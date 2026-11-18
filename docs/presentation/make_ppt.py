"""Builds CrowdGuardAI_Presentation.pptx (12 slides, 16:9). Run: python make_ppt.py"""
import os

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(HERE, "..", "screenshots")
OUT = os.path.join(HERE, "CrowdGuardAI_Presentation.pptx")

NAVY = RGBColor(0x14, 0x24, 0x3B)
INK = RGBColor(0x22, 0x2B, 0x36)
GREY = RGBColor(0x5F, 0x6B, 0x7A)
LIGHT = RGBColor(0xF2, 0xF4, 0xF7)
ACCENT = RGBColor(0x0E, 0x7C, 0x86)   # teal
RED = RGBColor(0xC0, 0x39, 0x2B)
GREEN = RGBColor(0x1E, 0x84, 0x49)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = prs.slide_width, prs.slide_height
TOTAL = 12


def text(slide, x, y, w, h, runs, size=20, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """runs: str or list of paragraphs; a paragraph may be (text, dict(size=,bold=,color=))."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    paras = runs if isinstance(runs, list) else [runs]
    for i, p in enumerate(paras):
        t, opt = (p if isinstance(p, tuple) else (p, {}))
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.alignment = align
        par.space_after = Pt(opt.get("after", 6))
        r = par.add_run()
        r.text = t
        r.font.name = FONT
        r.font.size = Pt(opt.get("size", size))
        r.font.bold = opt.get("bold", bold)
        r.font.color.rgb = opt.get("color", color)
    return tb


def bullets(slide, x, y, w, h, items, size=20, gap=10):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        head, body = (it if isinstance(it, tuple) else (None, it))
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.space_after = Pt(gap)
        r0 = par.add_run()
        r0.text = "\u25AA  "
        r0.font.size = Pt(size)
        r0.font.color.rgb = ACCENT
        if head:
            r1 = par.add_run()
            r1.text = head + "  "
            r1.font.bold = True
            r1.font.size = Pt(size)
            r1.font.name = FONT
            r1.font.color.rgb = INK
        r2 = par.add_run()
        r2.text = body
        r2.font.size = Pt(size)
        r2.font.name = FONT
        r2.font.color.rgb = INK
    return tb


def box(slide, x, y, w, h, fill=LIGHT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    return s


def picture(slide, name, x, y, max_w, max_h):
    p = slide.shapes.add_picture(os.path.join(SHOTS, name), x, y)
    ratio = min(max_w / p.width, max_h / p.height)
    p.width, p.height = int(p.width * ratio), int(p.height * ratio)
    p.left = int(x + (max_w - p.width) / 2)
    p.line.color.rgb = RGBColor(0xC9, 0xD0, 0xD8)
    p.line.width = Pt(0.75)
    return p


def page(title, n, kicker=None, notes=None):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.18), H)
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
    if kicker:
        text(s, Inches(0.6), Inches(0.35), Inches(10), Inches(0.4), kicker.upper(), size=13, color=ACCENT, bold=True)
    text(s, Inches(0.6), Inches(0.62), Inches(12), Inches(0.9), title, size=32, color=NAVY, bold=True)
    line = s.shapes.add_connector(1, Inches(0.6), Inches(1.45), Inches(12.75), Inches(1.45))
    line.line.color.rgb = RGBColor(0xD5, 0xDB, 0xE1)
    text(s, Inches(0.6), Inches(7.0), Inches(8), Inches(0.35), "CrowdGuard AI  |  Saurav Mathur", size=11, color=GREY)
    text(s, Inches(11.8), Inches(7.0), Inches(1), Inches(0.35), f"{n} / {TOTAL}", size=11, color=GREY, align=PP_ALIGN.RIGHT)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def table(slide, x, y, w, rows, col_w, size=16, header_fill=NAVY, highlight_row=None, row_h=0.45):
    t = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, Inches(row_h) * len(rows)).table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for r in t.rows:
        r.height = Inches(row_h)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.text = str(val)
            c.margin_left = c.margin_right = Inches(0.1)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            f = (p.runs[0] if p.runs else p.add_run()).font
            f.name, f.size = FONT, Pt(size)
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = header_fill
                f.bold, f.color.rgb = True, WHITE
            else:
                c.fill.fore_color.rgb = RGBColor(0xE6, 0xF3, 0xF4) if i == highlight_row else (LIGHT if i % 2 else WHITE)
                f.color.rgb = INK
                f.bold = i == highlight_row
    return t


# ── 1. Title ───────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
strip = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(5.05), W, Inches(0.08))
strip.fill.solid(); strip.fill.fore_color.rgb = ACCENT; strip.line.fill.background()
text(s, Inches(0.9), Inches(1.5), Inches(11.5), Inches(1.2), "CrowdGuard AI", size=54, color=WHITE, bold=True)
text(s, Inches(0.9), Inches(2.65), Inches(11.5), Inches(1.4),
     "Crowd threat monitoring and deepfake media verification in one system",
     size=26, color=RGBColor(0xC8, 0xE6, 0xE9))
text(s, Inches(0.9), Inches(5.4), Inches(7), Inches(1.4), [
    ("Saurav Mathur  ·  [Enrolment No.]", {"size": 20, "bold": True, "color": WHITE}),
    ("Guide: [Supervisor Name], [Designation]", {"size": 16, "color": RGBColor(0xC8, 0xD3, 0xDE)}),
])
text(s, Inches(7.4), Inches(5.4), Inches(5.2), Inches(1.4), [
    ("[School Name]", {"size": 16, "color": RGBColor(0xC8, 0xD3, 0xDE)}),
    ("Rashtriya Raksha University  ·  [Month Year]", {"size": 16, "color": RGBColor(0xC8, 0xD3, 0xDE)}),
], align=PP_ALIGN.RIGHT)
s.notes_slide.notes_text_frame.text = "Introduce yourself and the project in one sentence: one tool that tells you what is happening in crowd footage and whether the footage itself is real."

# ── 2. Problem ─────────────────────────────────────────────────────────
s = page("Why this project?", 2, "The problem",
         "Give a concrete example: a fake clip of an armed crowd can send police to the wrong place.")
bullets(s, Inches(0.6), Inches(1.8), Inches(6.6), Inches(4.8), [
    ("Video is the main evidence", "in public-order incidents: CCTV, phone clips and social media."),
    ("Two questions, not one:", "what is happening in the scene, and can we trust the footage at all?"),
    ("Fakes are now easy to make:", "diffusion models and face-swap tools produce images people cannot tell apart."),
    ("Existing tools answer only one question:", "crowd analytics trusts every feed; deepfake detectors ignore the scene."),
], size=20, gap=14)
box(s, Inches(7.7), Inches(1.9), Inches(5.0), Inches(4.4), fill=RGBColor(0xFD, 0xF1, 0xEF))
text(s, Inches(8.0), Inches(2.15), Inches(4.4), Inches(4), [
    ("Example", {"size": 14, "bold": True, "color": RED}),
    ("A clip of an \u201carmed man in a crowd\u201d spreads on WhatsApp.", {"size": 20, "bold": True}),
    ("If it is real, responders must move fast.", {"size": 18, "color": GREY}),
    ("If it is AI-generated, acting on it wastes resources and can cause panic.", {"size": 18, "color": GREY}),
    ("CrowdGuard AI checks both before anyone acts.", {"size": 18, "bold": True, "color": ACCENT}),
])

# ── 3. Objectives ──────────────────────────────────────────────────────
s = page("What I set out to build", 3, "Objectives")
objs = [
    ("1", "Count and track people, detect guns and knives, score crowd risk"),
    ("2", "Detect AI-generated images and face swaps with several published models"),
    ("3", "Calibrate and combine the models using labelled data, not guesswork"),
    ("4", "Say \u201cinconclusive\u201d when a check fails, never a silent \u201cauthentic\u201d"),
    ("5", "Show evidence an operator can read: scores, heatmaps, keyframes"),
    ("6", "Combine both results into one incident verdict in a web console"),
]
for i, (num, t) in enumerate(objs):
    col, row = i % 2, i // 2
    x, y = Inches(0.6 + col * 6.2), Inches(1.85 + row * 1.6)
    box(s, x, y, Inches(5.9), Inches(1.35))
    text(s, x + Inches(0.25), y, Inches(0.8), Inches(1.35), num, size=40, color=ACCENT, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + Inches(1.1), y, Inches(4.6), Inches(1.35), t, size=19, anchor=MSO_ANCHOR.MIDDLE)

# ── 4. Architecture ────────────────────────────────────────────────────
s = page("How the system is built", 4, "Architecture",
         "Point out that the face model runs in its own process: if it fails, the rest of the system keeps working and the result says the face check did not run.")
def node(x, y, w, h, title, sub, fill=LIGHT, edge=ACCENT):
    box(s, Inches(x), Inches(y), Inches(w), Inches(h), fill=fill, line=edge)
    text(s, Inches(x), Inches(y + 0.08), Inches(w), Inches(h - 0.1), [
        (title, {"size": 18, "bold": True, "color": NAVY, "after": 2}),
        (sub, {"size": 14, "color": GREY}),
    ], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
def arrow(x1, y1, x2, y2):
    c = s.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = GREY; c.line.width = Pt(2)
    c.line._get_or_add_ln().append(c.line._get_or_add_ln().makeelement(
        "{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd", {"type": "triangle"}))
node(4.65, 1.75, 4.0, 0.95, "Web console", "Next.js 14  ·  port 3000")
node(4.65, 3.05, 4.0, 0.95, "FastAPI backend", "PyTorch on RTX 4050  ·  port 8000")
node(0.7, 4.45, 3.7, 1.15, "Crowd module", "YOLOv8 person/gun/knife, MobileNetV3 check, risk engine")
node(4.8, 4.45, 3.7, 1.15, "Forensics module", "Community Forensics, B-Free, CLIP, calibrated fusion")
node(8.9, 4.45, 3.7, 1.15, "Face worker", "GenD CLIP ViT-L/14  ·  port 8001")
node(3.4, 5.95, 6.5, 0.85, "Investigation engine  \u2192  one incident verdict", "", fill=RGBColor(0xE6, 0xF3, 0xF4))
arrow(6.65, 2.7, 6.65, 3.05)
arrow(5.6, 4.0, 2.55, 4.45); arrow(6.65, 4.0, 6.65, 4.45); arrow(8.5, 5.02, 8.9, 5.02)
arrow(2.55, 5.6, 4.6, 5.95); arrow(6.65, 5.6, 6.65, 5.95)

# ── 5. Crowd surveillance ──────────────────────────────────────────────
s = page("Crowd surveillance", 5, "Module 1",
         "The weapon weights come from my earlier SentinelVision project. Mention the tracker fix: sparse frames made ByteTrack drop every new box, so detection now runs frame by frame.")
picture(s, "04_crowd_surveillance_tracking.png", Inches(0.6), Inches(1.75), Inches(7.6), Inches(4.9))
bullets(s, Inches(8.5), Inches(1.8), Inches(4.4), Inches(5), [
    ("Detection:", "YOLOv8 trained on person, gun and knife."),
    ("Second check:", "MobileNetV3 confirms the weapon type."),
    ("Risk score:", "5 points per person, 70\u2013100 per weapon, +25 for fast movement."),
    ("Levels:", "SAFE \u2192 ELEVATED \u2192 ALERT \u2192 CRITICAL."),
    ("Feed check:", "the same forensics verdict is shown on every frame."),
], size=17, gap=12)

# ── 6. Deepfake forensics ──────────────────────────────────────────────
s = page("Deepfake forensics", 6, "Module 2",
         "Explain why four models: each one catches different fakes. Community Forensics is strongest on diffusion images, GenD on face swaps.")
picture(s, "02_forensics_verdict.png", Inches(7.4), Inches(1.75), Inches(5.5), Inches(4.0))
rows = [["Model", "Looks at", "Source"],
        ["Community Forensics", "Whole image", "CVPR 2025"],
        ["B-Free (DINOv2)", "Whole image, native size", "CVPR 2025"],
        ["UniversalFakeDetect", "Whole image (CLIP)", "CVPR 2023"],
        ["GenD", "Each detected face", "WACV 2026"]]
table(s, Inches(0.6), Inches(1.8), Inches(6.5), rows, [2.4, 2.5, 1.6], size=15)
bullets(s, Inches(0.6), Inches(4.35), Inches(6.5), Inches(2.4), [
    "Each score is calibrated, then a fitted logistic model combines them.",
    "Evidence shown: ELA map, attention map, noise residual, EXIF / C2PA.",
    "Three modes: Auto, Deepfake face, AI-generated.",
], size=17, gap=8)

# ── 7. What went wrong ─────────────────────────────────────────────────
s = page("Version 1 gave poor results \u2014 so I measured every part", 7, "Testing",
         "This is the most important slide. Say that the published models were fine; the integration was wrong. Measuring each part on labelled data found all of it within a day.")
probs = [
    ("Pixel detector was dead", "The NPR weights gave about the same score for every image. AUC \u2248 0.50, i.e. chance."),
    ("Wrong face model", "Docs said GenD, but a generic classifier was loaded. It flagged 43% of real faces as fake."),
    ("No calibration", "Raw scores from different models were compared directly."),
    ("Hand-set rules", "One rule capped the score at 0.36, which forced some fakes to \u201creal\u201d."),
    ("Silent failure", "If the face worker timed out, the result said \u201cAuthentic Face\u201d."),
    ("Untrusted benchmark", "Old test file: 82% fake, unknown source. Discarded."),
]
for i, (h, b) in enumerate(probs):
    col, row = i % 2, i // 2
    x, y = Inches(0.6 + col * 6.2), Inches(1.8 + row * 1.65)
    box(s, x, y, Inches(5.9), Inches(1.45), fill=RGBColor(0xFD, 0xF1, 0xEF))
    text(s, x + Inches(0.25), y + Inches(0.12), Inches(5.4), Inches(1.3), [
        (h, {"size": 18, "bold": True, "color": RED, "after": 3}),
        (b, {"size": 15}),
    ])

# ── 8. Fix: data and calibration ───────────────────────────────────────
s = page("The fix: real models, real data, fitted weights", 8, "Method",
         "Stress that the test half was never used for fitting, and that each image was also tested as a JPEG copy to mimic shared images.")
bullets(s, Inches(0.6), Inches(1.8), Inches(6.2), Inches(5), [
    ("Replaced", "NPR with official B-Free (outputs match the authors' demo exactly)."),
    ("Installed", "the real GenD weights for faces."),
    ("Collected", "1,031 labelled public images from GitHub and Hugging Face."),
    ("Scored twice:", "original file and JPEG quality 85 copy."),
    ("Split 50/50", "by file hash: fit on one half, report on the other."),
    ("Fitted", "Platt calibration per model + logistic fusion."),
], size=18, gap=11)
rows = [["Dataset", "Real", "Fake"],
        ["Defactify (COCO vs 5 generators)", "250", "250"],
        ["DeepFakeFace (faces)", "250", "249"],
        ["GenD crops (FF++, Celeb-DF)", "12", "20"],
        ["Total", "512", "519"]]
table(s, Inches(7.1), Inches(1.9), Inches(5.7), rows, [3.5, 1.1, 1.1], size=15, highlight_row=4)
text(s, Inches(7.1), Inches(4.4), Inches(5.7), Inches(2), [
    ("Plus 189 Celeb-DF v2 test videos (100 real, 89 fake) prepared for video testing.", {"size": 15, "color": GREY}),
])

# ── 9. Results ─────────────────────────────────────────────────────────
s = page("Results on 1,010 unseen images", 9, "Evaluation",
         "Read the headline numbers slowly: fakes caught 62 to 79, real correct 76 to 96. Then admit the two weak spots on the chart: DALL-E 3 and InsightFace swaps.")
rows = [["", "Version 1", "Now"],
        ["Fakes detected", "62%", "79%"],
        ["Real images correct", "76%", "96%"],
        ["Balanced accuracy", "69%", "88%"],
        ["False alarms on real faces", "43%", "3%"],
        ["AUC", "\u2013", "0.943"]]
table(s, Inches(0.6), Inches(1.8), Inches(5.3), rows, [2.9, 1.2, 1.2], size=17, highlight_row=3)
cd = CategoryChartData()
cd.categories = ["SD 2.1", "SDXL", "SD 3", "DALL-E 3", "Midjourney", "InsightFace", "Inpainting"]
cd.add_series("Version 1", (96, 52, 36, 13, 4, 55, 99))
cd.add_series("Now", (100, 96, 95, 40, 65, 46, 100))
ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(6.2), Inches(1.7), Inches(6.7), Inches(4.6), cd).chart
ch.has_title = True
ch.chart_title.text_frame.text = "Fakes detected by generator (%)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(15)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(13)
ch.value_axis.maximum_scale = 100
ch.value_axis.has_major_gridlines = False
ch.value_axis.tick_labels.font.size = Pt(12)
ch.category_axis.tick_labels.font.size = Pt(12)
for ser, col in zip(ch.series, (RGBColor(0xB8, 0xC2, 0xCC), ACCENT)):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
plot = ch.plots[0]
plot.gap_width = 60
plot.has_data_labels = True
plot.data_labels.font.size = Pt(10)
plot.data_labels.font.color.rgb = INK
text(s, Inches(0.6), Inches(5.0), Inches(5.3), Inches(1.6), [
    ("Weak spots: DALL-E 3 (40%) and InsightFace face swaps (46%).", {"size": 15, "color": RED, "bold": True}),
    ("JPEG copies scored the same as originals, so normal compression does not break it.", {"size": 15, "color": GREY}),
])

# ── 10. Dual investigation ─────────────────────────────────────────────
s = page("Dual investigation: both checks, one verdict", 10, "Module 3",
         "Demo this live if possible: upload the CCTV clip and let the three stages tick through. A typical video takes about 7 to 15 seconds.")
picture(s, "06_dual_investigation_pipeline.png", Inches(0.6), Inches(1.7), Inches(6.0), Inches(2.95))
picture(s, "05_dual_investigation_result.png", Inches(6.85), Inches(1.7), Inches(6.0), Inches(2.95))
rows = [["Crowd says", "Media says", "Verdict"],
        ["Weapon", "Fake", "Fabricated threat  (CRITICAL)"],
        ["No weapon", "Fake", "Media manipulation  (HIGH)"],
        ["Weapon", "Real", "Verified incident  (CRITICAL)"],
        ["> 15 people", "Real", "Verified crowd surge  (ELEVATED)"]]
table(s, Inches(0.6), Inches(4.85), Inches(12.2), rows, [2.6, 2.6, 7.0], size=14, row_h=0.38)

# ── 11. Limitations & future ───────────────────────────────────────────
s = page("Limitations and next steps", 11, "Honest assessment")
text(s, Inches(0.6), Inches(1.75), Inches(6), Inches(0.5), "Limitations", size=22, color=RED, bold=True)
bullets(s, Inches(0.6), Inches(2.35), Inches(5.9), Inches(4.4), [
    "Video accuracy not measured yet (test set is ready).",
    "DALL-E 3 and face swaps still caught less than half the time.",
    "Weapon model finds weapons in only some frames.",
    "Calibration data covers 5 generators and 2 face datasets.",
    "B-Free licence: non-profit use only.",
], size=18, gap=12)
text(s, Inches(6.9), Inches(1.75), Inches(6), Inches(0.5), "Review 2 plan", size=22, color=GREEN, bold=True)
bullets(s, Inches(6.9), Inches(2.35), Inches(5.9), Inches(4.4), [
    "Run the 189-video Celeb-DF test and fit a video model.",
    "Add Flux, SD 3.5 and WhatsApp-compressed images.",
    "Optical-flow crowd motion and live RTSP cameras.",
    "Signed PDF incident report and incident database.",
    "Use operator feedback for periodic recalibration.",
], size=18, gap=12)

# ── 12. Conclusion ─────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
text(s, Inches(0.9), Inches(0.8), Inches(11.5), Inches(0.8), "What I learned", size=34, color=WHITE, bold=True)
points = [
    "Good published models can still give bad results if they are wired in wrongly.",
    "Measuring each part on labelled data beat tuning rules by hand: 69% \u2192 88% balanced accuracy.",
    "A security tool must say \u201cI could not check\u201d instead of guessing \u201cauthentic\u201d.",
]
for i, p in enumerate(points):
    y = Inches(1.9 + i * 1.15)
    num = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.9), y, Inches(0.7), Inches(0.7))
    num.fill.solid(); num.fill.fore_color.rgb = ACCENT; num.line.fill.background()
    num.text_frame.text = str(i + 1)
    r = num.text_frame.paragraphs[0].runs[0]; r.font.size = Pt(22); r.font.bold = True; r.font.color.rgb = WHITE
    num.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    text(s, Inches(1.9), y - Inches(0.05), Inches(10.6), Inches(0.9), p, size=22,
         color=RGBColor(0xE4, 0xEA, 0xF0), anchor=MSO_ANCHOR.MIDDLE)
strip = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(5.6), W, Inches(0.06))
strip.fill.solid(); strip.fill.fore_color.rgb = ACCENT; strip.line.fill.background()
text(s, Inches(0.9), Inches(5.85), Inches(6), Inches(0.9), "Thank you  \u2014  questions?", size=30, color=WHITE, bold=True)
text(s, Inches(6.5), Inches(5.95), Inches(6.1), Inches(0.9),
     "github.com/sauravmathur02/CrowdGuard-AI", size=16, color=RGBColor(0xC8, 0xE6, 0xE9), align=PP_ALIGN.RIGHT)
text(s, Inches(11.8), Inches(7.0), Inches(1), Inches(0.35), f"12 / {TOTAL}", size=11,
     color=RGBColor(0x9A, 0xA8, 0xB8), align=PP_ALIGN.RIGHT)

prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides))
