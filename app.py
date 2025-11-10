# app.py — LLM-assisted Math Chatbot (multimodal-ready)
# Run: streamlit run app.py

import os, re, ast
from typing import Optional, Dict, Any
import streamlit as st

# Optional deps (app runs even if missing)
try:
    import sympy as sp
except Exception:
    sp = None
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# ---- Optional LLM client (OpenAI SDK v1 style) ----
OPENAI_AVAILABLE = False
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # pick any chat model you have
try:
    from openai import OpenAI
    _OPENAI_KEY = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None)
    if _OPENAI_KEY:
        client = OpenAI(api_key=_OPENAI_KEY)
        OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---------------- UI ----------------
st.set_page_config(page_title="LLM Math Chatbot", page_icon="🧮", layout="wide")
st.title("🧮 Math Chatbot — LLM assisted (multimodal-ready)")
st.caption("Understands natural questions via an LLM, computes exactly with SymPy. Upload image/audio/PDF if you want; I’ll route automatically.")

# ---------------- State ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Helpers ----------------
def to_latex(obj) -> str:
    if sp is None:
        return str(obj)
    try:
        return sp.latex(obj)
    except Exception:
        return str(obj)

def infer_intent(text: str, has_image: bool, has_audio: bool, has_pdf: bool) -> str:
    t = (text or "").lower()
    if has_image or any(k in t for k in ["image", "picture", "photo", "screenshot", "diagram"]):
        return "image"
    if has_audio or any(k in t for k in ["audio", "voice", "lecture recording", "mp3", "wav", "m4a", "transcribe"]):
        return "audio"
    if has_pdf or any(k in t for k in ["slides", "pdf", "lecture notes", "handout", "deck"]):
        return "lecture"
    return "chat"

MATH_HELP = (
    "I can help with:\n"
    "• Calculus: limits, derivatives, partials, integrals, series\n"
    "• Algebra: solve equations/systems, simplify/factor/expand, inequalities\n"
    "• Linear algebra: matrices (det, rank, inverse), eigenvalues/vectors\n"
    "• Number theory: gcd/lcm, prime factorization, modular inverse\n"
    "Examples:\n"
    "- derivative of sin(x)^2\n"
    "- integrate x^2 from 0 to 1\n"
    "- limit (1+1/n)^n as n->oo\n"
    "- solve {x+y=3, x-y=1}\n"
    "- simplify (x^2 - 1)/(x-1)\n"
    "- matrix eigenvalues [[1,2],[3,4]]\n"
)

# ---------- Lightweight regex (local) fallback intent+parse ----------
def detect_math_op_local(raw: str):
    t = (raw or "").lower().replace("’","'")
    # derivative
    m = re.search(r"(?:what(?:'|)s\s+the\s+)?(?:derivative|differentiate|d/dx)\s+(?:of\s+)?(.+)", t)
    if m: return "derivative", {"expr": m.group(1).strip()}
    # integral (optional bounds)
    m = re.search(r"(?:integral|integrate)\s+(?:of\s+)?(.+?)\s*(?:from\s+([^\s]+)\s+to\s+([^\s]+))?$", t)
    if m: return "integral", {"expr": m.group(1).strip(), "a": m.group(2), "b": m.group(3)}
    # limit
    m = re.search(r"limit\s*(.+?)\s*as\s*([a-zA-Z])\s*->\s*([^\s]+)", t)
    if m: return "limit", {"expr": m.group(1).strip(), "var": m.group(2), "to": m.group(3)}
    # solve
    m = re.search(r"(?:solve|roots|solution)\s+(.+)", t)
    if m: return "solve", {"expr": m.group(1).strip()}
    # simplify/factor/expand
    m = re.search(r"(simplify|factor|expand)\s+(.+)", t)
    if m: return m.group(1).lower(), {"expr": m.group(2).strip()}
    return None, {}

# ---------- LLM router+parser ----------
LLM_PARSE_SYS = (
    "You convert natural-language math questions into a JSON instruction for a CAS (SymPy). "
    "Return compact JSON with keys: op in {derivative,integral,limit,solve,simplify,factor,expand,none}, "
    "expr (string SymPy-friendly), a (lower bound, optional), b (upper bound, optional), "
    "var (symbol for limit), to (target for limit). If you cannot parse, return op:'none'. "
    "Do not include any prose besides JSON."
)

def llm_parse_math(raw: str) -> Dict[str, Any]:
    """Use LLM to robustly parse messy user text into a CAS-ready plan."""
    if not OPENAI_AVAILABLE:
        return {"op":"none"}
    try:
        prompt = f"User: {raw}\nReturn JSON only."
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":LLM_PARSE_SYS},
                      {"role":"user","content":raw}],
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        # Extract JSON (tolerate leading/trailing text just in case)
        import json
        start = text.find("{"); end = text.rfind("}")
        obj = json.loads(text[start:end+1]) if start!=-1 and end!=-1 else json.loads(text)
        # normalize
        obj.setdefault("op","none")
        return obj
    except Exception:
        return {"op":"none"}

# ---------- LLM explainer (optional pretty steps) ----------
LLM_EXPLAIN_SYS = (
    "You are a math TA. Given a user's question and CAS result, explain steps clearly in 3-6 short bullets. "
    "Use LaTeX inline when helpful; keep it concise."
)
def llm_explain(user_q: str, result_text: str) -> Optional[str]:
    if not OPENAI_AVAILABLE:
        return None
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":LLM_EXPLAIN_SYS},
                      {"role":"user","content":f"Question: {user_q}\nCAS result: {result_text}"}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None

# ---------------- Math Engine (LLM-assisted) ----------------
def _parse_matrix(txt: str):
    try:
        m = ast.literal_eval(txt)
        return sp.Matrix(m) if sp is not None else None
    except Exception:
        return None

def do_sympy_compute(op: str, info: Dict[str, Any]) -> str:
    """Run the CAS operation with SymPy based on parsed plan."""
    if sp is None:
        return "SymPy isn't installed. Add `sympy` to requirements.txt."
    x, y, z, n, t = sp.symbols("x y z n t")

    if op == "derivative":
        expr = sp.sympify(info["expr"])
        return f"d/dx {expr} = {sp.simplify(sp.diff(expr, x))}"

    if op == "integral":
        expr = sp.sympify(info["expr"])
        a, b = info.get("a"), info.get("b")
        if a and b:
            val = sp.integrate(expr, (x, sp.sympify(a), sp.sympify(b)))
            return f"∫[{a},{b}] {expr} dx = {sp.simplify(val)}"
        else:
            return f"∫ {expr} dx = {sp.simplify(sp.integrate(expr, x))} + C"

    if op == "limit":
        var = sp.Symbol(info.get("var","x"))
        expr = sp.sympify(info["expr"])
        to_txt = info.get("to","oo")
        to_val = sp.oo if to_txt in ["oo","+inf","+infty","infinity"] else (-sp.oo if to_txt in ["-oo","-inf"] else sp.sympify(to_txt))
        res = sp.limit(expr, var, to_val)
        return f"limit {expr} as {var}->{to_txt} = {sp.simplify(res)}"

    if op == "solve":
        expr_txt = info["expr"]
        if "{" in expr_txt and "}" in expr_txt:
            inside = expr_txt[expr_txt.find("{")+1:expr_txt.rfind("}")]
            eqs = [e.strip() for e in inside.split(",")]
            symset, parsed = set(), []
            for e in eqs:
                L, R = e.split("=")
                parsed.append(sp.Eq(sp.sympify(L), sp.sympify(R)))
                symset |= sp.sympify(L).free_symbols | sp.sympify(R).free_symbols
            sol = sp.solve(parsed, list(symset))
            return f"Solutions: {sol}"
        if "=" in expr_txt:
            L, R = expr_txt.split("=", 1)
            sol = sp.solve(sp.Eq(sp.sympify(L), sp.sympify(R)))
        else:
            sol = sp.solve(sp.sympify(expr_txt))
        return f"Solutions: {sol}"

    if op in {"simplify","factor","expand"}:
        expr = sp.sympify(info["expr"])
        if op == "factor":  return f"factor({expr}) = {sp.factor(expr)}"
        if op == "expand":  return f"expand({expr}) = {sp.expand(expr)}"
        return f"simplify({expr}) = {sp.simplify(expr)}"

    return "I couldn't parse that.\n\n" + MATH_HELP

def math_engine(user_query: str, use_llm: bool) -> str:
    """
    1) Use LLM to parse op/expr if enabled; otherwise regex fallback.
    2) Execute with SymPy.
    3) Optionally add LLM explanation.
    """
    # 1) plan
    plan = {"op":"none"}
    if use_llm and OPENAI_AVAILABLE:
        plan = llm_parse_math(user_query)

    if plan.get("op") == "none":
        op, info = detect_math_op_local(user_query)
        plan = {"op": op or "none", **(info or {})}

    # 2) compute
    try:
        result = do_sympy_compute(plan.get("op","none"), plan)
    except Exception as e:
        result = f"Error while computing: {e}"

    # 3) optional explanation from LLM
    if use_llm and OPENAI_AVAILABLE:
        expl = llm_explain(user_query, result)
        if expl:
            return result + "\n\n**Explanation**\n" + expl
    return result

# ---------------- Upload Handlers (stubs) ----------------
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

# ---------------- Sidebar ----------------
st.sidebar.header("🧭 Router & Uploads")
img_file = st.sidebar.file_uploader("Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"])
aud_file = st.sidebar.file_uploader("Upload audio (WAV/MP3/M4A)", type=["wav", "mp3", "m4a"])
pdf_file = st.sidebar.file_uploader("Upload lecture PDF", type=["pdf"])

st.sidebar.markdown("---")
st.sidebar.subheader("LLM settings")
use_llm = st.sidebar.checkbox("Use LLM assistance (better parsing + explanations)", value=True)
if use_llm and not OPENAI_AVAILABLE:
    st.sidebar.warning("Set OPENAI_API_KEY env var or Streamlit secret to enable LLM.")

st.sidebar.markdown("---")
st.sidebar.subheader("Examples")
st.sidebar.code("derivative of x^2", language="text")
st.sidebar.code("integrate x^2 from 0 to 1", language="text")
st.sidebar.code("limit (1+1/n)^n as n->oo", language="text")
st.sidebar.code("solve {x+y=3, x-y=1}", language="text")
st.sidebar.code("matrix eigenvalues [[1,2],[3,4]]", language="text")

# ---------------- Chat history ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# ---------------- Input & Routing ----------------
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
        reply = math_engine(user_text, use_llm=use_llm)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)
