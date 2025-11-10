# app.py — Interactive Math Chatbot (multimodal-ready)
# Run locally: streamlit run app.py

import io, os, re
from typing import Optional
import streamlit as st

# Optional deps (app still runs if missing)
try:
    import sympy as sp
except Exception:
    sp = None

try:
    import PyPDF2  # lightweight PDF text extractor
except Exception:
    PyPDF2 = None

# ---------- UI ----------
st.set_page_config(page_title="Math Chatbot (Multimodal)", page_icon="🧮", layout="wide")
st.title("🧮 Math Chatbot — multimodal-ready")
st.caption("Ask math questions, or upload an image/audio/PDF; the assistant will route automatically.")

# ---------- State ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{'role': 'user'|'assistant', 'content': str}]

# ---------- Router ----------
def infer_intent(text: str, has_image: bool, has_audio: bool, has_pdf: bool) -> str:
    t = (text or "").lower()
    if has_image or any(k in t for k in ["image", "picture", "photo", "screenshot", "diagram"]):
        return "image"
    if has_audio or any(k in t for k in ["audio", "voice", "lecture recording", "mp3", "wav", "m4a", "transcribe"]):
        return "audio"
    if has_pdf or any(k in t for k in ["slides", "pdf", "lecture notes", "handout", "deck"]):
        return "lecture"
    return "chat"

# ---------- Math Engine ----------
MATH_HELP = (
    "I can help with: limits, derivatives, integrals, equation solving, simplifying, matrices.\n"
    "Examples:\n"
    "- differentiate sin(x)^2\n"
    "- integrate exp(-x^2) dx\n"
    "- limit (1+1/n)^n as n->oo\n"
    "- solve x^2 - 5x + 6 = 0\n"
    "- simplify (x^2 - 1)/(x-1)\n"
    "- matrix eigenvalues [[1,2],[3,4]]\n"
)

def _parse_matrix(txt: str):
    try:
        import ast
        m = ast.literal_eval(txt)
        return sp.Matrix(m) if sp is not None else None
    except Exception:
        return None

def math_engine(query: str) -> str:
    if sp is None:
        return "SymPy isn't installed here. Add `sympy` to requirements.txt to enable math solving."
    x, y, z, n = sp.symbols("x y z n")
    q = (query or "").strip()

    # Matrix (e.g., "matrix eigenvalues [[1,2],[3,4]]" or just "[[1,2],[3,4]]")
    if "matrix" in q.lower() or q.startswith("[["):
        mtxt_match = re.search(r"\[\[.*\]\]", q.replace("\n", " "))
        m = _parse_matrix(mtxt_match.group(0) if mtxt_match else q)
        if m is None:
            return "Couldn't parse the matrix. Try: matrix eigenvalues [[1,2],[3,4]]"
        if "eigen" in q.lower():
            return f"Eigenvalues: {m.eigenvals()}"
        if "det" in q.lower():
            return f"determinant = {m.det()}"
        if "rank" in q.lower():
            return f"rank = {m.rank()}"
        return f"Matrix {m.shape}:\n{m}"

    # Limit: "limit <expr> as x->oo"
    lim = re.search(r"limit\s*(.*)\s*as\s*([a-zA-Z])\s*->\s*([^\s]+)", q, re.I)
    if lim:
        expr_txt, var_txt, to_txt = lim.groups()
        var = sp.Symbol(var_txt)
        try:
            expr = sp.sympify(expr_txt)
            to_val = sp.oo if to_txt in ["oo","+inf","+infty","infinity"] else (-sp.oo if to_txt in ["-oo","-inf"] else sp.sympify(to_txt))
            res = sp.limit(expr, var, to_val)
            return f"limit {expr_txt} as {var_txt}->{to_txt} = {sp.simplify(res)}"
        except Exception as e:
            return f"Could not compute limit: {e}"

    # Derivative: "differentiate ...", "derivative ...", or "d/dx ..."
    if any(k in q.lower() for k in ["differentiate", "derivative", "d/dx", "derive"]):
        expr_txt = re.sub(r"(?i)(differentiate|derivative|d/dx|derive)\s*", "", q)
        try:
            expr = sp.sympify(expr_txt)
            res = sp.diff(expr, sp.Symbol("x"))
            return f"d/dx {expr} = {sp.simplify(res)}"
        except Exception as e:
            return f"Could not differentiate: {e}"

    # Integral: "integrate x^2 from 0 to 1" or "integrate sin(x)"
    if "integrate" in q.lower() or q.startswith("∫"):
        expr_txt = re.sub(r"(?i)integrate\s*", "", q)
        bounds = re.search(r"from\s+([^\s]+)\s+to\s+([^\s]+)$", expr_txt)
        try:
            if bounds:
                body = expr_txt[:bounds.start()].strip()
                a, b = bounds.groups()
                expr = sp.sympify(body)
                return f"∫[{a},{b}] {expr} dx = {sp.simplify(sp.integrate(expr, (sp.Symbol('x'), sp.sympify(a), sp.sympify(b))))}"
            expr = sp.sympify(expr_txt)
            return f"∫ {expr} dx = {sp.simplify(sp.integrate(expr, sp.Symbol('x')))} + C"
        except Exception as e:
            return f"Could not compute integral: {e}"

    # Solve equation: "solve x^2 - 5x + 6 = 0"
    if any(k in q.lower() for k in ["solve", "roots", "solution"]):
        expr_txt = re.sub(r"(?i)(solve|roots|solution)\s*", "", q)
        try:
            if "=" in expr_txt:
                left, right = expr_txt.split("=", 1)
                sol = sp.solve(sp.Eq(sp.sympify(left), sp.sympify(right)))
            else:
                sol = sp.solve(sp.sympify(expr_txt))
            return f"Solutions: {sol}"
        except Exception as e:
            return f"Could not solve equation: {e}"

    # Simplify/factor/expand
    if any(k in q.lower() for k in ["simplify", "factor", "expand"]):
        try:
            expr_txt = re.sub(r"(?i)(simplify|factor|expand)\s*", "", q)
            expr = sp.sympify(expr_txt)
            if "factor" in q.lower():  return f"factor({expr}) = {sp.factor(expr)}"
            if "expand" in q.lower():  return f"expand({expr}) = {sp.expand(expr)}"
            return f"simplify({expr}) = {sp.simplify(expr)}"
        except Exception as e:
            return f"Could not process expression: {e}"

    # Fallback
    try:
        expr = sp.sympify(q)
        return f"Parsed: {expr}\nSimplified: {sp.simplify(expr)}"
    except Exception:
        return "I couldn't parse that.\n\n" + MATH_HELP

# ---------- Upload handlers (stubs) ----------
def handle_image(file) -> str:
    name = getattr(file, "name", "uploaded_image")
    return f"I received an image: {name}. (OCR not configured in this demo.)"

def handle_audio(file) -> str:
    name = getattr(file, "name", "uploaded_audio")
    return f"I received an audio file: {name}. (Transcription not configured in this demo.)"

def handle_pdf(file) -> str:
    name = getattr(file, "name", "uploaded.pdf")
    if PyPDF2 is None:
        return f"Got a PDF ({name}), but PyPDF2 is not installed."
    try:
        reader = PyPDF2.PdfReader(file)
        texts = []
        for p in reader.pages[:5]:
            texts.append((p.extract_text() or "").strip())
        preview = "\n---\n".join(texts)
        if not preview.strip():
            return f"Could not extract text from {name} (might be scanned images)."
        return f"Extracted text preview from {name}:\n\n{preview[:1500]}"
    except Exception as e:
        return f"Error reading PDF: {e}"

# ---------- Sidebar ----------
st.sidebar.header("🧭 Router & Uploads")
img_file = st.sidebar.file_uploader("Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"])
aud_file = st.sidebar.file_uploader("Upload audio (WAV/MP3/M4A)", type=["wav", "mp3", "m4a"])
pdf_file = st.sidebar.file_uploader("Upload lecture PDF", type=["pdf"])

st.sidebar.markdown("---")
st.sidebar.subheader("Examples")
st.sidebar.code("differentiate sin(x)^2", language="text")
st.sidebar.code("integrate x^2 from 0 to 1", language="text")
st.sidebar.code("limit (1+1/n)^n as n->oo", language="text")
st.sidebar.code("solve x^2 - 5x + 6 = 0", language="text")

# ---------- Chat history ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# ---------- Input ----------
user_text = st.chat_input("Type your math question or instruction…")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    intent = infer_intent(
        user_text,
        has_image=img_file is not None,
        has_audio=aud_file is not None,
        has_pdf=pdf_file is not None
    )

    if intent == "image":
        reply = "It sounds like you want to use an image. Upload one in the sidebar." if img_file is None else handle_image(img_file)
    elif intent == "audio":
        reply = "It sounds like you want to use audio. Upload one in the sidebar." if aud_file is None else handle_audio(aud_file)
    elif intent == "lecture":
        reply = "It sounds like you want to process lecture notes. Upload a PDF in the sidebar." if pdf_file is None else handle_pdf(pdf_file)
    else:
        reply = math_engine(user_text)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)
