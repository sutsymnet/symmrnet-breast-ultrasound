"""
input_gate.py - Colour-Doppler screening for the SymMRNet deployment prototype
=============================================================================
วางไฟล์นี้ที่ repo root ข้าง streamlit_app.py และ symmrnet_core.py

เหตุผล
------
SymMRNet ฝึกบนภาพ B-mode ขาวดำ (Kaggle DS03.3) เท่านั้น
ภาพ Color Doppler มีสีทับบนเนื้อก้อน ทำให้ข้อมูล echo จริงถูกแทนที่
decode_and_resize() จะแปลงเป็น grayscale ก่อนเข้าโมเดลอยู่แล้ว สีจึงกลาย
เป็นก้อนเทากลาง ๆ ที่ไม่ใช่เนื้อเยื่อจริง -> out-of-distribution input
ผลทำนายเชื่อถือไม่ได้ และ "คาดเดาทิศทางความผิดพลาดไม่ได้" ด้วย
(ไม่ได้แปลว่าจะสูงเกินจริงเสมอไป)

ทำไม "เตือน" ไม่ใช่ "บล็อก"
--------------------------
ต้นทุนของความผิดพลาดสองแบบไม่เท่ากัน
  - เตือนภาพดีผิด      -> ผู้ใช้กดยืนยันผ่าน เสียเวลา 2 วินาที
  - ปล่อย Doppler ผ่าน -> ได้ผลทำนายที่เชื่อไม่ได้ โดยไม่มีใครรู้ตัว
จึงตั้งเกณฑ์ให้ไวไว้ก่อน แล้วให้คนตัดสินขั้นสุดท้าย

ไม่ต้องเพิ่ม dependency
----------------------
ใช้ numpy + pillow ซึ่ง requirements.txt มีอยู่แล้ว
ถ้าเครื่องมี OpenCV ติดตั้งอยู่ จะใช้เส้นทางที่แม่นกว่าโดยอัตโนมัติ

ประสิทธิภาพที่วัดได้ (อ่านก่อนใช้)
--------------------------------
คาลิเบรตกับภาพ 316 ภาพ (DatasetA/B/C ของ EC cohort)
                       จับ Doppler   เตือนภาพดีเกินจำเป็น
    numpy (ค่าเริ่มต้น)    30/34 (88%)     47/282 (17%)
    OpenCV (ถ้ามี)        33/34 (97%)     35/282 (12%)

*** ข้อจำกัดสำคัญ: ตัวเลขนี้วัดบน thumbnail 160x160 ไม่ใช่ภาพเต็ม ***
    การย่อภาพทำให้ flow จุดเล็กจางหาย ค่าจริงบนภาพเต็มน่าจะดีกว่านี้
    แต่ยังไม่มีใครวัด -> อย่าอ้างตัวเลขนี้ในวิทยานิพนธ์
    จนกว่าจะรัน calibrate() บนภาพความละเอียดเต็มพร้อม ground truth

สิ่งที่ต้อง "ผ่าน" (ไม่ใช่ Doppler)
  - ภาพขาวดำปกติ
  - ภาพที่มี caliper สี cyan      (annotation ของเครื่อง)
  - ภาพที่มีกรอบ ROI สีเขียว       (เปิดโหมด Doppler ค้าง แต่ไม่มี flow)
  - ภาพที่อมม่วง/อมน้ำเงินทั้งภาพ  (color cast ของจอ)

สิ่งที่ต้อง "เตือน"
  - flow สีแดง/ส้ม/เหลือง/น้ำเงิน เป็นหย่อมบนเนื้อภาพ
=============================================================================
"""

import io

import numpy as np
from PIL import Image

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# ---- เกณฑ์ (คาลิเบรตแล้ว - อ่านหมายเหตุด้านบนก่อนแก้) ----
SAT_MIN     = 12      # ความอิ่มสีขั้นต่ำ หลังลบ global colour cast
DOM_MIN     = 10      # ช่องสีต้องเด่นกว่าช่องอื่นเท่านี้ ถึงจะนับว่าเป็นสีนั้น
BLOCK       = 8       # ขนาดบล็อกที่ใช้ตรวจว่าสีเกาะกลุ่ม (ไม่ใช่ noise กระจาย)
BLOCK_MIN   = 2       # ในบล็อก 8x8 ต้องมีพิกเซล flow อย่างน้อยเท่านี้
FLOW_FRAC   = 0.0003  # สัดส่วนพิกเซล flow ต่อพื้นที่ภาพจริง
MIN_BLOCKS  = 2       # จำนวนบล็อกขั้นต่ำ

ACTIVE_LUM  = 20      # ความสว่างขั้นต่ำที่นับว่าเป็นพื้นที่ภาพ (ตัดขอบดำ)


def _to_rgb(img):
    """รับได้ทั้ง bytes / PIL.Image / numpy array -> คืน numpy RGB uint8"""
    if isinstance(img, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img))
    if isinstance(img, Image.Image):
        img = np.array(img.convert("RGB"))
    img = np.asarray(img)
    if img.ndim == 2:                       # grayscale
        img = np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 4:  # RGBA
        img = img[:, :, :3]
    return img.astype(np.uint8)


def analyze(img):
    """
    วิเคราะห์ภาพ คืน dict ของ metric (ใช้ debug / เก็บ log / ทำรายงานได้)
    """
    try:
        rgb = _to_rgb(img)
    except Exception as exc:                 # noqa: BLE001
        return {"error": f"อ่านภาพไม่ได้: {exc}"}

    if rgb.size == 0:
        return {"error": "ภาพว่าง"}

    a = rgb.astype(np.int16)
    R, G, B = a[:, :, 0].copy(), a[:, :, 1], a[:, :, 2].copy()

    lum = 0.299 * R + 0.587 * G + 0.114 * B
    act = lum > ACTIVE_LUM                   # พื้นที่ภาพจริง ตัดขอบดำ
    n_act = int(act.sum())
    if n_act < 100:
        return {"error": "ไม่พบพื้นที่ภาพ"}

    # ---- ลบ global colour cast (ภาพอมม่วงจะได้ไม่ถูกเข้าใจผิด) ----
    cast_rg = float(np.median((R - G)[act]))
    cast_bg = float(np.median((B - G)[act]))
    R = R - cast_rg
    B = B - cast_bg

    # ---- แยกทิศทางสี: นับเฉพาะ flow ตัด cyan/เขียวทิ้ง ----
    mx = np.maximum(np.maximum(R, G), B)
    mn = np.minimum(np.minimum(R, G), B)
    saturated = (mx - mn) > SAT_MIN

    cyan   = (np.minimum(G, B) - R) > DOM_MIN   # caliper ของเครื่อง -> ผ่าน
    green  = (G - np.maximum(R, B)) > DOM_MIN   # กรอบ ROI          -> ผ่าน
    red    = (R - np.maximum(G, B)) > DOM_MIN   # flow
    blue   = (B - np.maximum(R, G)) > DOM_MIN   # flow
    yellow = (np.minimum(R, G) - B) > DOM_MIN   # flow ความเร็วสูง

    flow = saturated & act & (red | blue | yellow) & ~cyan & ~green

    if _HAS_CV2:
        # เส้นทางแม่นกว่า: opening ทิ้ง noise + นับหย่อมด้วย connected components
        f8 = flow.astype(np.uint8)
        f8 = cv2.morphologyEx(f8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(f8, 8)
        min_area = max(3, int(0.00005 * n_act))
        blobs = [i for i in range(1, n) if st[i, 4] >= min_area]
        flow_px = int(sum(st[i, 4] for i in blobs))
        n_blocks = len(blobs)
        frac = flow_px / n_act
        need_frac, need_blocks, backend = 0.0003, 1, "opencv"
    else:
        # เส้นทาง numpy ล้วน: นับบล็อก 8x8 ที่มีสีเกาะกลุ่ม
        h, w = flow.shape
        hh, ww = h // BLOCK * BLOCK, w // BLOCK * BLOCK
        if hh == 0 or ww == 0:
            return {"error": "ภาพเล็กเกินไป"}
        blk = flow[:hh, :ww].reshape(hh // BLOCK, BLOCK,
                                     ww // BLOCK, BLOCK).sum((1, 3))
        n_blocks = int((blk >= BLOCK_MIN).sum())
        flow_px = int(flow.sum())
        frac = flow_px / n_act
        need_frac, need_blocks, backend = FLOW_FRAC, MIN_BLOCKS, "numpy"

    return {
        "backend": backend,
        "flow_frac": round(frac, 6),
        "flow_px": flow_px,
        "flow_blobs": n_blocks,
        "active_px": n_act,
        "cast_rg": round(cast_rg, 1),
        "cast_bg": round(cast_bg, 1),
        "is_doppler": bool(frac >= need_frac and n_blocks >= need_blocks),
    }


WARNING_TEXT = (
    "⚠️ **ตรวจพบสัญญาณสีในภาพ — อาจเป็นภาพ Color Doppler**\n\n"
    "ระบบนี้ได้รับการฝึกและตรวจสอบความถูกต้องกับ **ภาพ B-mode ขาวดำ** เท่านั้น "
    "ภาพ Color Doppler มีสีทับบนเนื้อก้อน ทำให้ข้อมูลอัลตราซาวด์จริงถูกแทนที่ "
    "ผลทำนายจึงอาจคลาดเคลื่อนโดยไม่สามารถคาดเดาทิศทางได้\n\n"
    "หากภาพนี้เป็นภาพ B-mode ปกติ (เช่น มีเพียง caliper สีหรือกรอบ ROI) "
    "สามารถกดยืนยันเพื่อดำเนินการต่อได้"
)


def check_image_bytes(image_bytes):
    """
    ประตูหลัก -> (needs_confirm, message, metrics)

      needs_confirm = False  ผ่านได้เลย
      needs_confirm = True   ควรเตือน และให้ผู้ใช้ยืนยันก่อนทำนาย

    ฟังก์ชันนี้ไม่ "บล็อก" — การตัดสินขั้นสุดท้ายเป็นของผู้ใช้
    """
    m = analyze(image_bytes)
    if "error" in m:
        return True, f"อ่านภาพไม่ได้: {m['error']}", m
    if m["is_doppler"]:
        return True, WARNING_TEXT, m
    return False, "", m


# alias เผื่อเรียกด้วยชื่อเดิม
check_image = check_image_bytes


def calibrate(root, labels_csv=None):
    """
    รันทุกภาพใน root แล้วคืน DataFrame ของ metric
    ถ้ามี labels_csv (คอลัมน์ rel_path, is_doppler) จะคำนวณ
    อัตราการจับได้ / อัตราเตือนเกินให้ด้วย

    ใช้สำหรับคาลิเบรตซ้ำบนภาพความละเอียดเต็ม ก่อนอ้างตัวเลขในวิทยานิพนธ์
    วิธีสร้าง labels: เปิด contact sheet ไล่ดูด้วยตา ทำเครื่องหมายภาพที่เป็น
    Color Doppler จริง (ประมาณ 30 นาทีสำหรับ 316 ภาพ)
    """
    import glob
    import os

    import pandas as pd

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
             if p.lower().endswith(exts)]
    rows = []
    for p in files:
        with open(p, "rb") as fh:
            m = analyze(fh.read())
        m["rel_path"] = os.path.relpath(p, root)
        rows.append(m)
    df = pd.DataFrame(rows)

    if labels_csv and os.path.exists(labels_csv):
        lab = pd.read_csv(labels_csv)
        df = df.merge(lab[["rel_path", "is_doppler"]], on="rel_path",
                      how="left", suffixes=("_pred", "_true"))
        d = df[df.is_doppler_true.notna()]
        pos = int((d.is_doppler_true == 1).sum())
        neg = int((d.is_doppler_true == 0).sum())
        tp = int((d.is_doppler_pred & (d.is_doppler_true == 1)).sum())
        fp = int((d.is_doppler_pred & (d.is_doppler_true == 0)).sum())
        if pos:
            print(f"จับ Doppler ได้        : {tp}/{pos}  ({100*tp/pos:.0f}%)")
        if neg:
            print(f"เตือนภาพดีเกินจำเป็น   : {fp}/{neg}  ({100*fp/neg:.0f}%)")
        print("\nปรับ FLOW_FRAC ขึ้น = เตือนน้อยลง แต่พลาด Doppler มากขึ้น")
    return df


if __name__ == "__main__":
    print(__doc__)
    print(f"OpenCV available: {_HAS_CV2}")
