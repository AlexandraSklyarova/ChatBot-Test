# app.py — Math Chatbot (multimodal-ready) with natural-language parsing
# Run: streamlit run app.py

import io, os, re, ast
from typing import Optional, List
import streamlit as st

# Optional deps (the app still runs if missing)
try:
    import sympy as sp
except Exception:
    sp = None
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# ---------------- UI ----------------
st.set_page_config(page_title="Math Chatbot (Multimodal)", page_icon="🧮", layout="wide")
st.title("Math Chatbot")
st.caption("Ask math questions, or upload an image/audio/PDF; the assistant routes automatically.")

# ---------------- State ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{'role': 'user'|'assistant', 'content': str}]

# ---------------- Helpers ----------------
def to_latex(obj) -> str:
    if sp is None:
        return str(obj)
    try:
        return sp.latex(obj)
    except Exception:
        return str(obj)

def extract_expression(text: str) -> str:
    """
    Pull the math expression out of a natural-language prompt.
    e.g., "what's the derivative of x^2?" -> "x^2"
    """
    t = (text or "").strip()
    t = t.replace("’", "'").lower()

    # quick phrase normalizations
    t = t.replace("what is", "whatis").replace("what's", "whats")

    patterns = [
        r"derivative of (.*)",
        r"differentiate (.*)",
        r"integral of (.*)",
        r"integrate (.*)",
        r"limit of (.*)",
        r"whatis (.*)",
        r"whats (.*)",
        r"solve (.*)",
        r"simplify (.*)",
        r"expand (.*)",
        r"factor (.*)",
    ]
    for p in patterns:
        m = re.search(p, t)
        if m:
            expr = m.group(1)
            # strip common tails
            expr = re.sub(r"( with respect to .*| wrt .*| please| thanks|[?.!])+$", "", expr).strip()
            # handle "as x->..." for limits (keep it in full query; math_engine handles)
            return expr if not p.startswith("limit") else text
    return text
    
def detect_math_op(raw: str):
    """
    Return (op, payload) where op in {'derivative','integral','limit','solve','simplify',None}.
    payload is a dict with fields used by the op.
    """
    t = (raw or "").lower().replace("’","'")
    # derivative
    m = re.search(r"(?:what(?:'|)s\s+the\s+)?(?:derivative|differentiate|d/dx)\s+(?:of\s+)?(.+)", t)
    if m:
        return "derivative", {"expr": m.group(1).strip()}
    # integral (definite: "... from a to b")
    m = re.search(r"(?:integral|integrate)\s+(?:of\s+)?(.+?)\s*(?:from\s+([^\s]+)\s+to\s+([^\s]+))?$", t)
    if m:
        return "integral", {"expr": m.group(1).strip(), "a": m.group(2), "b": m.group(3)}
    # limit: "limit (expr) as x->value"
    m = re.search(r"limit\s*(.+?)\s*as\s*([a-zA-Z])\s*->\s*([^\s]+)", t)
    if m:
        return "limit", {"expr": m.group(1).strip(), "var": m.group(2), "to": m.group(3)}
    # solve (systems allowed later)
    m = re.search(r"(?:solve|roots|solution)\s+(.+)", t)
    if m:
        return "solve", {"expr": m.group(1).strip()}
    # simplify/factor/expand
    m = re.search(r"(simplify|factor|expand)\s+(.+)", t)
    if m:
        return m.group(1).lower(), {"expr": m.group(2).strip()}
    return None, {}


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
    "• Calculus: limits, derivatives (partial), integrals, series (Taylor/Maclaurin)\n"
    "• Algebra: solve equations/systems, simplify/factor/expand, inequalities\n"
    "• Linear algebra: matrices (det, rank, inverse), eigenvalues/vectors\n"
    "• Number theory: gcd/lcm, prime factorization, modular inverse\n"
    "Examples:\n"
    "- differentiate sin(x)^2\n"
    "- integrate x^2 from 0 to 1\n"
    "- limit (1+1/n)^n as n->oo\n"
    "- solve {x+y=3, x-y=1}\n"
    "- simplify (x^2 - 1)/(x-1)\n"
    "- matrix eigenvalues [[1,2],[3,4]]\n"
)

# ---------------- Math Engine ----------------
def _parse_matrix(txt: str):
    try:
        m = ast.literal_eval(txt)
        return sp.Matrix(m) if sp is not None else None
    except Exception:
        return None

def math_engine(query: str) -> str:
    if sp is None:
        return "SymPy isn't installed. Add `sympy` to requirements.txt."

    # 1) Detect op on the RAW text
    op, info = detect_math_op(query)

    # 2) Then extract the expression (for nicer NL handling)
    cleaned = extract_expression(query)
    q = (cleaned or "").strip()

    x, y, z, n, t = sp.symbols("x y z n t")

    # ---------- DERIVATIVE ----------
    if op == "derivative":
        try:
            expr = sp.sympify(info["expr"])
            return f"d/dx {expr} = {sp.simplify(sp.diff(expr, x))}"
        except Exception as e:
            return f"Could not differentiate: {e}"

    # ---------- INTEGRAL ----------
    if op == "integral":
        try:
            expr = sp.sympify(info["expr"])
            a, b = info.get("a"), info.get("b")
            if a and b:
                val = sp.integrate(expr, (x, sp.sympify(a), sp.sympify(b)))
                return f"∫[{a},{b}] {expr} dx = {sp.simplify(val)}"
            else:
                return f"∫ {expr} dx = {sp.simplify(sp.integrate(expr, x))} + C"
        except Exception as e:
            return f"Could not compute integral: {e}"

    # ---------- LIMIT ----------
    if op == "limit":
        try:
            var = sp.Symbol(info["var"])
            expr = sp.sympify(info["expr"])
            to_txt = info["to"]
            to_val = sp.oo if to_txt in ["oo","+inf","+infty","infinity"] else (-sp.oo if to_txt in ["-oo","-inf"] else sp.sympify(to_txt))
            res = sp.limit(expr, var, to_val)
            return f"limit {expr} as {var}->{to_txt} = {sp.simplify(res)}"
        except Exception as e:
            return f"Could not compute limit: {e}"

    # ---------- SOLVE / SIMPLIFY / FACTOR / EXPAND ----------
    if op == "solve":
        try:
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
        except Exception as e:
            return f"Could not solve: {e}"

    if op in {"simplify","factor","expand"}:
        try:
            expr = sp.sympify(info["expr"])
            if op == "factor":  return f"factor({expr}) = {sp.factor(expr)}"
            if op == "expand":  return f"expand({expr}) = {sp.expand(expr)}"
            return f"simplify({expr}) = {sp.simplify(expr)}"
        except Exception as e:
            return f"Could not process expression: {e}"

    # ---------- (rest of your handlers: matrix, number theory, fallback, etc.) ----------
    # keep your existing matrix/number-theory blocks here...
    # Fallback:
    try:
        expr = sp.sympify(q)
        return f"Parsed: ${to_latex(expr)}$\nSimplified: ${to_latex(sp.simplify(expr))}$"
    except Exception:
        return "I couldn't parse that.\n\n" + MATH_HELP


    # ---- Matrix helpers ----
    if "matrix" in q.lower() or q.strip().startswith("[["):
        mtxt = re.search(r"\[\[.*\]\]", q.replace("\n", " "))
        mat = _parse_matrix(mtxt.group(0) if mtxt else q)
        if mat is None:
            return "Couldn't parse the matrix. Try: matrix eigenvalues [[1,2],[3,4]]"
        lower = q.lower()
        if "eigen" in lower:
            return f"Eigenvalues: {mat.eigenvals()}"
        if "det" in lower:
            return f"determinant = {mat.det()}"
        if "rank" in lower:
            return f"rank = {mat.rank()}"
        if "inverse" in lower or "inv" in lower:
            return f"inverse:\n{mat.inv()}"
        return f"Matrix {mat.shape}:\n{mat}"

    # ---- Limit: 'limit <expr> as x->oo' ----
    lim = re.search(r"limit\s*(.*)\s*as\s*([a-zA-Z])\s*->\s*([^\s]+)", q, re.I)
    if lim and sp is not None:
        expr_txt, var_txt, to_txt = lim.groups()
        var = sp.Symbol(var_txt)
        try:
            expr = sp.sympify(expr_txt)
            to_val = sp.oo if to_txt in ["oo","+inf","+infty","infinity"] else (-sp.oo if to_txt in ["-oo","-inf"] else sp.sympify(to_txt))
            res = sp.limit(expr, var, to_val)
            return f"limit {expr_txt} as {var_txt}->{to_txt} = {sp.simplify(res)}"
        except Exception as e:
            return f"Could not compute limit: {e}"

    # ---- Derivative: 'differentiate ...' or natural '... derivative of ...' ----
    if any(k in q.lower() for k in ["differentiate", "derivative", "d/dx", "derive"]):
        expr_txt = re.sub(r"(?i)(differentiate|derivative|d/dx|derive)\s*", "", q)
        try:
            expr = sp.sympify(expr_txt)
            res = sp.diff(expr, x)
            return f"d/dx {expr} = {sp.simplify(res)}"
        except Exception as e:
            return f"Could not differentiate: {e}"

    # If the cleaned query is a bare expression, try derivative when user asked naturally
    if re.search(r"\bderivative\b|\bd/dx\b|\bdifferentiate\b", (query or "").lower()):
        try:
            expr = sp.sympify(extract_expression(query))
            res = sp.diff(expr, x)
            return f"d/dx {expr} = {sp.simplify(res)}"
        except Exception as e:
            return f"Could not differentiate: {e}"

    # ---- Integral: 'integrate ...' or 'integral of ...' ----
    if "integrate" in q.lower() or q.lower().startswith("∫") or "integral" in q.lower():
        expr_txt = re.sub(r"(?i)(integrate|integral of)\s*", "", q)
        bounds = re.search(r"from\s+([^\s]+)\s+to\s+([^\s]+)$", expr_txt)
        try:
            if bounds:
                body = expr_txt[:bounds.start()].strip()
                a, b = bounds.groups()
                expr = sp.sympify(body)
                val = sp.integrate(expr, (x, sp.sympify(a), sp.sympify(b)))
                return f"∫[{a},{b}] {expr} dx = {sp.simplify(val)}"
            expr = sp.sympify(expr_txt)
            return f"∫ {expr} dx = {sp.simplify(sp.integrate(expr, x))} + C"
        except Exception as e:
            return f"Could not compute integral: {e}"

    # ---- Solve equations / systems / inequalities ----
    if any(k in q.lower() for k in ["solve", "roots", "solution", "inequality"]):
        try:
            if "inequality" in q.lower():
                expr_txt = re.sub(r"(?i)solve\s*inequality\s*", "", q)
                sol = sp.solve_univariate_inequality(sp.sympify(expr_txt), x, relational=True)
                return f"Solution set: ${to_latex(sol)}$"
            expr_txt = re.sub(r"(?i)(solve|roots|solution)\s*", "", q)
            if "{" in expr_txt and "}" in expr_txt:  # system like {x+y=3, x-y=1}
                inside = expr_txt[expr_txt.find("{")+1:expr_txt.rfind("}")]
                eqs = [e.strip() for e in inside.split(",")]
                symset = set()
                parsed = []
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
        except Exception as e:
            return f"Could not solve: {e}"

    # ---- Simplify / factor / expand ----
    if any(k in q.lower() for k in ["simplify", "factor", "expand"]):
        try:
            expr_txt = re.sub(r"(?i)(simplify|factor|expand)\s*", "", q)
            expr = sp.sympify(expr_txt)
            if "factor" in q.lower():  return f"factor({expr}) = {sp.factor(expr)}"
            if "expand" in q.lower():  return f"expand({expr}) = {sp.expand(expr)}"
            return f"simplify({expr}) = {sp.simplify(expr)}"
        except Exception as e:
            return f"Could not process expression: {e}"

    # ---- Fallback: just try to parse and simplify ----
    try:
        expr = sp.sympify(q)
        return f"Parsed: ${to_latex(expr)}$\nSimplified: ${to_latex(sp.simplify(expr))}$"
    except Exception:
        return "I couldn't parse that.\n\n" + MATH_HELP

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
st.sidebar.subheader("Examples")
st.sidebar.code("differentiate sin(x)^2", language="text")
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
        reply = math_engine(user_text)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)
