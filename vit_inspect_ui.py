# =====================================================================================
#  ViT-Inspect  |  Interactive Gradio UI   (run as a NEW CELL at the END of the notebook)
#  Needs: notebook Steps 1-5 executed in this runtime (functions) + Step 7 done (saved banks).
#  Theme: Jet Black #0A0A0A | Cerulean Blue #007BA7 | Neon Rose #FF2D95
# =====================================================================================
# !pip -q install "gradio>=5,<6"      # <- run once in its own cell first
import random, html
import gradio as gr

JET, BLUE, ROSE = "#0A0A0A", "#007BA7", "#FF2D95"
RNG = random.Random(7)
CMAP = plt.get_cmap("jet")                       # same colour map as the notebook heat maps
BANKS, TEST_CACHE = {}, {}                       # caches so the app stays fast

# ------------------------------- 1. BACKEND (model logic) -------------------------------
def available_categories():
    """Only categories whose memory bank was saved in the notebook (Step 6/7)."""
    return [c for c in ALL_CATEGORIES if os.path.exists(f"{OUT_DIR}/banks/vit_{c}.pt")] or ALL_CATEGORIES

def get_bank(tag, cat):
    """Load the saved memory bank + threshold once, then reuse."""
    if (tag, cat) not in BANKS:
        path = f"{OUT_DIR}/banks/{tag}_{cat}.pt"
        if not os.path.exists(path):
            raise gr.Error(f"No saved memory bank for {tag.upper()} / {cat}. Run the notebook for this category first.")
        ck = torch.load(path, map_location=DEVICE); ck["bank"] = ck["bank"].float()
        BANKS[(tag, cat)] = ck
    return BANKS[(tag, cat)]

@torch.no_grad()
def infer(pil, cat, tag):
    """PIL image -> anomaly map + score (same pipeline as notebook Step 5)."""
    ck = get_bank(tag, cat)
    img = pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
    x = ((x - MEAN) / STD).unsqueeze(0)
    amap = make_anomaly_maps(EXTRACTORS[tag], x, ck["bank"])[0]
    return dict(img=img, amap=amap, score=float(amap.max()), thr=ck["threshold"],
                lo=ck["score_min"], hi=ck["score_max"], tag=tag)

def verdict_html(st, sens):
    """Big result card: blue = NORMAL, neon rose = DEFECTIVE, with a score-vs-threshold meter."""
    thr = st["thr"] * sens
    bad = st["score"] >= thr
    pct = min(100, st["score"] / (2 * thr) * 100)          # threshold marker sits at 50 %
    return (f'<div class="verdict {"bad" if bad else "ok"}"><div class="vw">{"DEFECTIVE" if bad else "NORMAL"}</div>'
            f'<div class="vs">{st["tag"].upper()} · score {st["score"]:.2f} · threshold {thr:.2f}</div>'
            f'<div class="meter"><div class="fill" style="width:{pct:.0f}%"></div><div class="mark"></div></div></div>')

def make_views(st, sens, opacity):
    """Heat-map overlay (opacity slider) and predicted regions (sensitivity slider)."""
    amap, base = st["amap"], np.asarray(st["img"])
    norm = np.clip((amap - st["lo"]) / (st["hi"] - st["lo"] + 1e-8), 0, 1)
    heat = (CMAP(norm)[..., :3] * 255).astype(np.uint8)
    blend = (base * (1 - opacity) + heat * opacity).astype(np.uint8)
    seg = base.copy(); pred = amap > st["thr"] * sens
    seg[pred] = (0.4 * seg[pred] + 0.6 * np.array([255, 45, 149])).astype(np.uint8)
    return Image.fromarray(blend), Image.fromarray(seg)

# ------------------------------- 2. UI EVENT FUNCTIONS -------------------------------
def analyse(img, cat, model):
    if img is None:
        raise gr.Error("Upload an image or press 🎲 Random test image")
    return infer(img, cat, model)

def render(st, sens, opacity):
    """Re-draw results from the cached state, so sliders update LIVE without re-running the model."""
    if st is None:
        return [gr.update()] * 5
    blend, seg = make_views(st, sens, opacity)
    path = f"{OUT_DIR}/heatmap_result.png"; blend.save(path)
    return verdict_html(st, sens), st["img"], blend, seg, path

def random_image(cat):
    """Pick a random official test image (normal or defective) from the dataset."""
    if cat not in TEST_CACHE:
        TEST_CACHE[cat] = MVTecDataset(DATA_ROOT, cat, "test", IMG_SIZE).samples
    return Image.open(RNG.choice(TEST_CACHE[cat])["img"]).convert("RGB")

def compare(img, cat):
    """Same image through ViT and CNN -> two verdicts + two heat maps."""
    if img is None:
        raise gr.Error("Upload an image or press 🎲 Random test image")
    out = []
    for tag in ["vit", "cnn"]:
        if tag not in EXTRACTORS:
            out += ["<div class='note'>CNN baseline was not run (RUN_CNN_BASELINE = False).</div>", None]; continue
        st = infer(img, cat, tag); out += [verdict_html(st, 1.0), make_views(st, 1.0, 0.5)[0]]
    return out

def dataset_gallery(cat):
    """One normal image + one example per defect type (mask in neon rose)."""
    tr = MVTecDataset(DATA_ROOT, cat, "train", IMG_SIZE); te = MVTecDataset(DATA_ROOT, cat, "test", IMG_SIZE)
    items, seen = [(tr.load_pil(tr.samples[0]["img"]), "normal (train)")], set()
    for s in te.samples:
        if s["type"] != "good" and s["type"] not in seen:
            seen.add(s["type"]); im = np.array(te.load_pil(s["img"]))
            m = np.asarray(Image.open(s["mask"]).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 0
            im[m] = (0.4 * im[m] + 0.6 * np.array([255, 45, 149])).astype(np.uint8)
            items.append((Image.fromarray(im), s["type"]))
    return items

def load_results():
    """Tables/plots produced in notebook Steps 8-11."""
    f = f"{OUT_DIR}/results_all.csv"
    if not os.path.exists(f):
        return None, None, None
    r = pd.read_csv(f)
    def pv(m):
        t = r.pivot(index="category", columns="backbone", values=m); t.loc["MEAN"] = t.mean()
        return t.round(4).reset_index()
    abl = f"{OUT_DIR}/ablation_layers.csv"
    return pv("image_auroc"), pv("pixel_auroc"), (pd.read_csv(abl) if os.path.exists(abl) else None)

# ------------------------------- 3. STYLE (CSS + theme) -------------------------------
CSS = f"""
body{{background:radial-gradient(900px 420px at 90% -8%,rgba(255,45,149,.20),transparent),
 radial-gradient(900px 480px at -8% 0%,rgba(0,123,167,.28),transparent),{JET}!important;background-attachment:fixed!important}}
.gradio-container{{background:transparent!important;max-width:1240px!important;margin-left:auto!important;margin-right:auto!important;padding:16px 20px!important}}
.hero{{display:flex;align-items:center;justify-content:space-between;gap:24px;padding:26px 30px;margin-bottom:8px;
 border:1px solid {BLUE};border-radius:20px;background:linear-gradient(120deg,rgba(0,123,167,.18),rgba(255,45,149,.10));
 box-shadow:0 0 30px rgba(0,123,167,.35)}}
.hero h1{{margin:0;font-size:46px;font-weight:800;color:#fff;letter-spacing:1px;text-shadow:0 0 18px {ROSE}}}
.hero h1 span{{color:{ROSE}}} .hero p{{margin:6px 0 12px;color:#c9d6dd;font-size:16px}}
.chip{{display:inline-block;margin:3px 6px 0 0;padding:4px 12px;border-radius:999px;border:1px solid {BLUE};color:#9fdcf0;font-size:12px}}
.chip.r{{border-color:{ROSE};color:#ffb3d9}}
.verdict{{border-radius:18px;padding:18px 22px;text-align:center;border:2px solid;margin-bottom:6px}}
.verdict.ok{{border-color:{BLUE};background:rgba(0,123,167,.15);box-shadow:0 0 28px rgba(0,123,167,.55)}}
.verdict.bad{{border-color:{ROSE};background:rgba(255,45,149,.13);box-shadow:0 0 28px rgba(255,45,149,.6)}}
.vw{{font-size:40px;font-weight:800;letter-spacing:4px;color:#fff}} .vs{{color:#cfd8dc;font-size:14px;margin-top:2px}}
.meter{{position:relative;height:10px;border-radius:6px;background:#1c1c1c;margin-top:12px;overflow:hidden}}
.ok .fill{{height:100%;background:{BLUE}}} .bad .fill{{height:100%;background:{ROSE}}}
.mark{{position:absolute;left:50%;top:-2px;width:3px;height:14px;background:#fff}}
.card{{background:rgba(255,255,255,.04);border:1px solid rgba(0,123,167,.5);border-radius:16px;padding:16px 18px;color:#dbe6ec;margin-bottom:10px}}
.card h3{{margin:0 0 6px;color:{ROSE}}} .card b{{color:#7fd3ee}}
.steps{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}
.step{{border-left:4px solid {ROSE};background:rgba(0,123,167,.10);border-radius:12px;padding:12px 14px;color:#dbe6ec}}
.step i{{color:{ROSE};font-style:normal;font-weight:800;font-size:22px}}
.note{{color:#ffb3d9;padding:12px}}
button.primary,.primary{{box-shadow:0 0 16px rgba(0,123,167,.7)!important;font-weight:700!important}}
button[role=tab][aria-selected=true]{{color:{ROSE}!important;border-color:{ROSE}!important}}
footer{{display:none!important}} .foot{{text-align:center;color:#78909c;padding:14px;font-size:13px}}
"""

_dark = dict(body_background_fill=JET, background_fill_primary="#101417", background_fill_secondary="#0d1114",
             block_background_fill="#101417", block_border_color="#1f3a46", body_text_color="#e6eef2",
             block_label_background_fill="#101417", block_label_text_color="#7fd3ee", input_background_fill="#0d1114",
             button_primary_background_fill=BLUE, button_primary_background_fill_hover=ROSE, button_primary_text_color="#fff",
             button_secondary_background_fill="#16232a", button_secondary_text_color="#9fdcf0",
             slider_color=ROSE, border_color_primary="#1f3a46", table_even_background_fill="#101417", table_odd_background_fill="#0d1a20")
theme = gr.themes.Base(font=[gr.themes.GoogleFont("Poppins"), "sans-serif"]).set(
    **_dark, **{k + "_dark": v for k, v in _dark.items()})

HERO = f"""<div class="hero"><div><h1>ViT<span>-</span>Inspect</h1>
<p>Unsupervised industrial defect detection with a Vision Transformer · MVTec AD</p>
<span class="chip">DINO ViT-B/16</span><span class="chip r">Memory bank + coreset</span><span class="chip">15 categories</span><span class="chip r">MAAI901</span></div>
<svg width="170" height="130" viewBox="0 0 170 130">{''.join(f'<rect x="{10+c*30}" y="{8+r*30}" width="26" height="26" rx="5" fill="{ROSE if (r,c)==(1,3) else BLUE}" opacity="{1 if (r,c)==(1,3) else .35}"/>' for r in range(4) for c in range(5))}</svg></div>"""

ABOUT_DATA = """<div class="card"><h3>What is MVTec AD?</h3>
<b>MVTec Anomaly Detection</b> is the standard benchmark for industrial visual inspection (Bergmann et al., CVPR 2019, MVTec Software GmbH).
It has <b>over 5,000 high-resolution images</b> in <b>15 categories</b>: <b>5 textures</b> (carpet, grid, leather, tile, wood) and
<b>10 objects</b> (bottle, cable, capsule, hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper).</div>
<div class="card"><h3>Why is it special?</h3>
Training set = <b>only normal images</b>. Test set = normal + defective images with <b>pixel-accurate ground-truth masks</b>
(scratches, cracks, dents, contamination, missing parts...). This matches real factories, where defects are rare and unpredictable.</div>
<div class="card"><h3>Folder structure</h3><code>category/train/good · category/test/&lt;defect_type&gt; · category/ground_truth/&lt;defect_type&gt;</code><br>
Source: Hugging Face <b>foersben/mvtec-ad</b> · License: <b>CC BY-NC-SA 4.0</b> (non-commercial use only).</div>"""

STEPS = [("1", "Patchify", "Image is cut into 16×16 patches (14×14 grid)."), ("2", "ViT features", "Frozen DINO ViT gives each patch a vector with global context via self-attention."),
         ("3", "Memory bank", "Patch features of normal images only, compressed by greedy coreset (10 %)."),
         ("4", "Nearest neighbour", "Each test patch: distance to closest normal patch. Far = anomalous."),
         ("5", "Heat map + score", "Distances form the anomaly map; its maximum is the image score.")]
METHOD = ('<div class="steps">' + "".join(f'<div class="step"><i>{n}</i> <b>{t}</b><br>{d}</div>' for n, t, d in STEPS) + "</div>"
          '<div class="card" style="margin-top:12px"><h3>Why a ViT?</h3>Self-attention lets every patch see the whole image, so features describe both local texture and context. '
          'Middle blocks (8, 11) keep fine local detail, which is where small defects live. A CNN baseline (WideResNet-50) uses the same pipeline for a fair comparison.</div>')

def data_usage_html():
    """Explains exactly which images are used for training and which for testing."""
    d = dataset_summary()
    tr, tg, td = int(d.train_normal.sum()), int(d.test_normal.sum()), int(d.test_defective.sum())
    return (f'<div class="card"><h3>How the data is used: train vs test</h3><div class="steps">'
            f'<div class="step"><i>{tr:,}</i><br><b>TRAIN: normal only</b><br>Defect-free images build the memory bank. The model never sees a defective image or a mask while training.</div>'
            f'<div class="step"><i>{tg:,}</i><br><b>TEST: normal</b><br>New defect-free images, used to check false alarms.</div>'
            f'<div class="step"><i>{td:,}</i><br><b>TEST: defective</b><br>Images with real defects + pixel masks, used only for evaluation (AUROC).</div></div>'
            f'<small style="color:#9fb3bd">Counts for the {len(d)} categories loaded in this session. The decision threshold is picked from test scores (Youden\'s J), '
            f'so the demo accuracy is slightly optimistic; AUROC does not depend on it.</small></div>')

def split_plot():
    """Bar chart: train normal / test normal / test defective per category."""
    d = dataset_summary(); x = np.arange(len(d)); w = 0.27
    fig, ax = plt.subplots(figsize=(11, 3.8)); fig.patch.set_facecolor(JET); ax.set_facecolor(JET)
    ax.bar(x - w, d.train_normal, w, color=BLUE, label="train (normal only)")
    ax.bar(x, d.test_normal, w, color="#7fd3ee", label="test normal")
    ax.bar(x + w, d.test_defective, w, color=ROSE, label="test defective")
    ax.set_xticks(x); ax.set_xticklabels(d.category, rotation=45, ha="right"); ax.tick_params(colors="white")
    ax.set_title("Images per category and split", color="white")
    for sp in ax.spines.values(): sp.set_color("#1f3a46")
    ax.legend(facecolor=JET, labelcolor="white", edgecolor="#1f3a46"); fig.tight_layout()
    return fig

# ------------------------------- 4. LAYOUT -------------------------------
cats = available_categories()
models = list(EXTRACTORS.keys())

with gr.Blocks(theme=theme, css=CSS, title="ViT-Inspect", js="() => document.body.classList.add('dark')") as demo:
    gr.HTML(HERO)
    with gr.Tabs():
        # ---------- Tab 1: Inspect ----------
        with gr.Tab("🔍 Inspect"):
            state = gr.State(None)
            with gr.Row():
                with gr.Column(scale=1):
                    cat = gr.Dropdown(cats, value=cats[0], label="Category")
                    model = gr.Radio(models, value="vit", label="Model (vit = Vision Transformer, cnn = baseline)")
                    img = gr.Image(type="pil", label="Upload product image", height=260)
                    with gr.Row():
                        run = gr.Button("🔍 Analyse", variant="primary"); rnd = gr.Button("🎲 Random test image")
                    sens = gr.Slider(0.5, 1.5, value=1.0, step=0.01, label="Sensitivity (threshold × value) · lower = stricter")
                    opa = gr.Slider(0, 1, value=0.5, step=0.05, label="Heat-map opacity")
                with gr.Column(scale=2):
                    verdict = gr.HTML("<div class='note'>Upload an image or press 🎲 to start.</div>")
                    with gr.Row():
                        o1 = gr.Image(label="Input", height=260); o2 = gr.Image(label="Anomaly heat map", height=260)
                        o3 = gr.Image(label="Predicted defect regions", height=260)
                    dl = gr.DownloadButton("⬇ Download heat map")
            outs = [verdict, o1, o2, o3, dl]
            run.click(analyse, [img, cat, model], state).then(render, [state, sens, opa], outs)
            rnd.click(random_image, cat, img).then(analyse, [img, cat, model], state).then(render, [state, sens, opa], outs)
            sens.change(render, [state, sens, opa], outs); opa.change(render, [state, sens, opa], outs)

        # ---------- Tab 2: Compare ----------
        with gr.Tab("⚖️ Compare ViT vs CNN"):
            with gr.Row():
                c2 = gr.Dropdown(cats, value=cats[0], label="Category")
                i2 = gr.Image(type="pil", label="Image", height=240)
                with gr.Column():
                    go = gr.Button("⚖️ Compare", variant="primary"); r2 = gr.Button("🎲 Random test image")
            with gr.Row():
                with gr.Column(): gr.Markdown("### 🟦 Vision Transformer"); v1 = gr.HTML(); m1 = gr.Image(label="ViT heat map")
                with gr.Column(): gr.Markdown("### 🌸 CNN baseline"); v2 = gr.HTML(); m2 = gr.Image(label="CNN heat map")
            go.click(compare, [i2, c2], [v1, m1, v2, m2])
            r2.click(random_image, c2, i2).then(compare, [i2, c2], [v1, m1, v2, m2])

        # ---------- Tab 3: Results ----------
        with gr.Tab("📊 Results"):
            t_img, t_pix, t_abl = load_results()
            if t_img is None:
                gr.HTML("<div class='note'>results_all.csv not found. Run notebook Step 7 first.</div>")
            else:
                gr.Markdown("### Image-level AUROC"); gr.Dataframe(t_img, interactive=False)
                gr.Markdown("### Pixel-level AUROC"); gr.Dataframe(t_pix, interactive=False)
                with gr.Row():
                    for f in ["image_auroc_comparison.png", "pixel_auroc_comparison.png"]:
                        if os.path.exists(f"{OUT_DIR}/{f}"): gr.Image(f"{OUT_DIR}/{f}", show_label=False)
                if t_abl is not None:
                    gr.Markdown("### Ablation: which ViT layers?"); gr.Dataframe(t_abl, interactive=False)

        # ---------- Tab 4: About the dataset ----------
        with gr.Tab("🗂️ About the Dataset"):
            gr.HTML(ABOUT_DATA)
            gr.HTML(data_usage_html()); gr.Plot(split_plot(), show_label=False)
            gr.Markdown("### Images per category"); gr.Dataframe(dataset_summary(), interactive=False)
            gc = gr.Dropdown(cats, value=cats[0], label="Explore a category")
            gal = gr.Gallery(label="Normal vs defect types (defect mask in neon rose)", columns=4, height=300)
            gc.change(dataset_gallery, gc, gal); demo.load(dataset_gallery, gc, gal)   # fill gallery on page load

        # ---------- Tab 5: About the method ----------
        with gr.Tab("🧠 About the Method"):
            gr.HTML(METHOD)

    gr.HTML('<div class="foot">ViT-Inspect · MAAI901 Deep Learning for Visual Data · built with PyTorch + Hugging Face + Gradio</div>')

demo.queue().launch(share=True, debug=True)     # share=True gives a public link you can open on your phone
