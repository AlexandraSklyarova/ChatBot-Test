# app.py — LLM-assisted Math Chatbot (multimodal-ready)
# Run: streamlit run app.py

import os, re, ast, difflib
from typing import Optional, Dict, Any
import streamlit as st

# env (.env local) — safe for local, harmless on Streamlit Cloud
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

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
st.title("Math Chatbot")
st.caption("Understands natural questions via an LLM, computes exactly with SymPy.")

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

# ---------------- Pretty rendering helpers ----------------
def _latex_clean(s: str) -> str:
    """Normalize common outputs to valid LaTeX."""
    # \[...\] -> $$...$$
    s = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", s, flags=re.S)
    # \to oo -> \to \infty
    s = re.sub(r"(?<=\\to)\s*oo", r" \\infty", s)
    s = re.sub(r"\\lim_\{([^}]*)\\to\s*oo\}", r"\\lim_{\1\\to \\infty}", s)
    # collapse accidental $$$$
    s = s.replace("$$$$", "$$")
    return s.strip()

def render_reply(reply: str):
    """Render mixed text/LaTeX nicely in chat."""
    if not isinstance(reply, str):
        st.markdown(str(reply)); return
    txt = _latex_clean(reply)

    # single display-math block
    if txt.startswith("$$") and txt.endswith("$$"):
        st.latex(txt.strip("$"))
        return

    # split into blocks: $$...$$ as LaTeX, rest as markdown
    blocks = re.split(r"(\$\$.*?\$\$)", txt, flags=re.S)
    for b in blocks:
        if not b:
            continue
        if b.startswith("$$") and b.endswith("$"):
            st.latex(b.strip("$"))
        elif b.startswith("$$") and b.endswith("$$"):
            st.latex(b.strip("$"))
        else:
            st.markdown(b)

# ---------------- Typo tolerance + robust parsing ----------------
_KEYWORDS = ["derivative", "differentiate", "d/dx",
             "integral", "integrate",
             "limit",
             "solve", "solution", "roots",
             "simplify", "factor", "expand"]

def fuzzy_fix_keyword(word: str, cutoff=0.75):
    m = difflib.get_close_matches(word, _KEYWORDS, n=1, cutoff=cutoff)
    return m[0] if m else word

def fuzzy_fix_ops(text: str) -> str:
    toks = text.split()
    if not toks: return text
    toks[0] = fuzzy_fix_keyword(toks[0].lower())
    if len(toks) > 1:
        toks[1] = fuzzy_fix_keyword(toks[1].lower())
    return " ".join(toks)

def insert_parens_after_func(expr: str) -> str:
    """
    sinx -> sin(x), sin 2x -> sin(2*x), ln x -> log(x), sqrtx -> sqrt(x)
    """
    expr = re.sub(r"\b(ln)\b", "log", expr)  # ln -> log
    for func in ["sin","cos","tan","cot","sec","csc","log","sqrt"]:
        expr = re.sub(rf"\b{func}\s*([a-zA-Z]\b)", rf"{func}(\1)", expr)     # sin x -> sin(x)
        expr = re.sub(rf"\b{func}([a-zA-Z])\b",     rf"{func}(\1)", expr)     # sinx  -> sin(x)
        expr = re.sub(rf"\b{func}\s*(\d+[a-zA-Z])", rf"{func}(\1)", expr)     # sin 2x -> sin(2x)
    return expr

def balance_parens(expr: str) -> str:
    opens = expr.count("("); closes = expr.count(")")
    if opens > closes:
        expr = expr + (")" * (opens - closes))
    return expr

from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
)
_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,  # 2x -> 2*x , (x+1)(x-1) -> ...
    convert_xor,                          # x^2 stays exponent
    function_exponentiation,              # sin^2(x) -> sin(x)**2
)

def try_parse(expr_txt: str):
    """Heuristics + SymPy parse_expr, with graceful fallback."""
    s = (expr_txt or "").strip().replace("’", "'")
    s = insert_parens_after_func(s)
    s = balance_parens(s)
    # Treat 'e' as Euler's number
    locals_map = {"e": sp.E} if sp else {}
    return parse_expr(s, transformations=_TRANSFORMS, evaluate=True, local_dict=locals_map)

def detect_math_op_local(raw: str):
    """
    Typo-tolerant intent detection for basic ops + 'explain' concept queries.
    Returns (op, info) where op∈{derivative,integral,limit,solve,simplify,factor,expand,explain,None}.
    """
    t = (raw or "").strip()
    t = fuzzy_fix_ops(t.lower().replace("’","'"))

    # ---- explain / what is / define / why queries ----
    m = re.search(r"^(explain|what\s+is|define|why\s+is|intuition\s+for)\s+(.+)$", t)
    if m:
        topic = m.group(2).strip(" ?.")
        return "explain", {"topic": topic}

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


# ---------- LLM router+parser (optional) ----------
LLM_PARSE_SYS = (
    "You convert natural-language math questions into a JSON instruction for a CAS (SymPy). "
    "Return compact JSON with keys: op in {derivative,integral,limit,solve,simplify,factor,expand,none}, "
    "expr (string SymPy-friendly), a (lower bound, optional), b (upper bound, optional), "
    "var (symbol for limit), to (target for limit). If you cannot parse, return op:'none'. "
    "Do not include any prose besides JSON."
)

def llm_parse_math(raw: str) -> Dict[str, Any]:
    if not OPENAI_AVAILABLE:
        return {"op":"none"}
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":LLM_PARSE_SYS},
                      {"role":"user","content":raw}],
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        import json
        start = text.find("{"); end = text.rfind("}")
        obj = json.loads(text[start:end+1]) if start!=-1 and end!=-1 else json.loads(text)
        obj.setdefault("op","none")
        return obj
    except Exception:
        return {"op":"none"}

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
# ---------- LLM concept explainer + offline fallback ----------
LLM_EXPLAIN_CONCEPT_SYS = (
    "You are a kind math TA. Explain the requested math concept clearly in 6-10 short bullets, "
    "with 1-2 tiny worked examples. Prefer symbols and LaTeX where helpful. "
    "Keep each bullet concise. Avoid unnecessary history or trivia."
)

# very short offline snippets for when LLM is unavailable
_OFFLINE_KB = {
    "derivative": r"""
**Derivative (intuition):**
- Measures *instantaneous rate of change* (slope of the tangent line).
- Definition: \( f'(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h} \).
- Rules: \( \frac{d}{dx}x^n = nx^{n-1}\), \( (\sin x)' = \cos x\), \( (e^x)'=e^x\).
- Example: \( f(x)=x^2 \Rightarrow f'(x)=2x\).
""",
    "integral": r"""
**Integral (intuition):**
- Accumulated quantity / signed area under curve.
- Indefinite: \( \int f(x)\,dx = F(x)+C\) where \(F' = f\).
- Definite: \( \int_a^b f(x)\,dx = F(b)-F(a)\).
- Example: \( \int x^2 dx = \frac{x^3}{3}+C\).
""",
    "limit": r"""
**Limit (intuition):**
- What \(f(x)\) approaches as \(x\) approaches a value.
- Notation: \( \lim_{x\to a} f(x) \).
- Example: \( \lim_{n\to\infty}\left(1+\frac{1}{n}\right)^n=e\).
""",
    "eigenvalues": r"""
**Eigenvalues / Eigenvectors:**
- \(A v = \lambda v\) with \(v\neq 0\). \(v\): eigenvector, \(\lambda\): eigenvalue.
- Solve \(\det(A-\lambda I)=0\) for \(\lambda\); then solve \((A-\lambda I)v=0\) for \(v\).
- Example \( \begin{bmatrix}1&2\\3&4\end{bmatrix}\): \(\lambda=\frac{5\pm\sqrt{33}}{2}\).
""",
    "rank": r"""
**Matrix Rank:**
- Number of linearly independent rows/columns.
- Equals pivot count in row-reduced echelon form.
- Dimension of the image of the linear map.
""",
}

def llm_explain_concept(topic: str) -> str:
    """Explain a concept with the LLM; fallback to a tiny offline note."""
    t = (topic or "").strip()
    if OPENAI_AVAILABLE:
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role":"system","content":LLM_EXPLAIN_CONCEPT_SYS},
                    {"role":"user","content":f"Explain this concept: {t}"}
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            pass  # fall through to offline
    # offline fallback
    key = t.lower().strip()
    for k in _OFFLINE_KB:
        if k in key:
            return _OFFLINE_KB[k]
    return f"Concept explanation unavailable offline for '{t}'. Enable LLM in the sidebar to get a detailed explanation."

# ---------------- Math Engine (LLM-assisted) ----------------
def _parse_matrix(txt: str):
    try:
        m = ast.literal_eval(txt)
        return sp.Matrix(m) if sp is not None else None
    except Exception:
        return None

def do_sympy_compute(op: str, info: Dict[str, Any]) -> str:
    """Run the CAS operation with SymPy based on parsed plan (pretty LaTeX)."""
    if sp is None:
        return "SymPy isn't installed. Add `sympy` to requirements.txt."
    x, y, z, n, t = sp.symbols("x y z n t")

    if op == "derivative":
        expr = try_parse(info["expr"])
        deriv = sp.simplify(sp.diff(expr, x))
        return f"$$\\frac{{d}}{{dx}}\\,{to_latex(expr)} = {to_latex(deriv)}$$"

    if op == "integral":
        expr = try_parse(info["expr"])
        a, b = info.get("a"), info.get("b")
        if a and b:
            aval = try_parse(a); bval = try_parse(b)
            val = sp.simplify(sp.integrate(expr, (x, aval, bval)))
            return f"$$\\int_{{{to_latex(aval)}}}^{{{to_latex(bval)}}} {to_latex(expr)}\\,dx = {to_latex(val)}$$"
        ant = sp.simplify(sp.integrate(expr, x))
        return f"$$\\int {to_latex(expr)}\\,dx = {to_latex(ant)} + C$$"

    if op == "limit":
        var = sp.Symbol(info.get("var","x"))
        expr = try_parse(info["expr"])
        to_txt = info.get("to","oo")
        to_val = sp.oo if to_txt in ["oo","+inf","+infty","infinity"] else \
                 -sp.oo if to_txt in ["-oo","-inf"] else try_parse(to_txt)
        pretty_to = r"\infty" if to_val == sp.oo else (r"-\infty" if to_val == -sp.oo else to_latex(to_val))
        res = sp.simplify(sp.limit(expr, var, to_val))
        return f"$$\\lim_{{{to_latex(var)}\\to {pretty_to}}} {to_latex(expr)} = {to_latex(res)}$$"

    if op == "solve":
        expr_txt = info["expr"]
        if "{" in expr_txt and "}" in expr_txt:
            inside = expr_txt[expr_txt.find("{")+1:expr_txt.rfind("}")]
            eqs = [e.strip() for e in inside.split(",")]
            symset, parsed = set(), []
            for e in eqs:
                L, R = e.split("=")
                Lp, Rp = try_parse(L), try_parse(R)
                parsed.append(sp.Eq(Lp, Rp))
                symset |= Lp.free_symbols | Rp.free_symbols
            sol = sp.solve(parsed, list(symset))
            return f"Solutions: {sol}"
        if "=" in expr_txt:
            L, R = expr_txt.split("=", 1)
            sol = sp.solve(sp.Eq(try_parse(L), try_parse(R)))
        else:
            sol = sp.solve(try_parse(expr_txt))
        return f"Solutions: {sol}"

    if op in {"simplify","factor","expand"}:
        expr = try_parse(info["expr"])
        if op == "factor":  return f"$$\\mathrm{{factor}}\\big({to_latex(expr)}\\big) = {to_latex(sp.factor(expr))}$$"
        if op == "expand":  return f"$$\\mathrm{{expand}}\\big({to_latex(expr)}\\big) = {to_latex(sp.expand(expr))}$$"
        return f"$$\\mathrm{{simplify}}\\big({to_latex(expr)}\\big) = {to_latex(sp.simplify(expr))}$$"

    return "I couldn't parse that.\n\n" + MATH_HELP

def math_engine(prompt: str, use_llm: bool = True) -> str:
    """
    1) Try to parse intent locally (typo tolerant).
    2) Execute with SymPy (robust parser).
    3) If that fails and LLM is enabled, ask LLM to parse/explain, then compute again if possible.
    """
    if sp is None:
        return "SymPy isn't installed. Add `sympy` to requirements.txt."

    # 1) Plan via local detector
    op, info = detect_math_op_local(prompt)
    plan = {"op": op or "none", **(info or {})}

    # 2) Compute with SymPy if possible
    if plan["op"] != "none":
        try:
            return do_sympy_compute(plan["op"], plan)
        except Exception:
            pass
    # 2.5) Bare expression fallback: try to parse/evaluate plain math like "9+10"
    try:
        expr = try_parse(prompt)
        val = sp.simplify(expr)
        # If it simplifies to a number, just show the value; otherwise show both
        if val.is_Number:
            return f"$${to_latex(val)}$$"
        # show "expr = simplified"
        if val != expr:
            return f"$${to_latex(expr)} = {to_latex(val)}$$"
        else:
            return f"$${to_latex(expr)}$$"
    except Exception:
        pass

    # 3) LLM-assisted parsing + explanation (optional)
    if use_llm and OPENAI_AVAILABLE:
        plan = llm_parse_math(prompt)
        if plan.get("op") != "none":
            try:
                result = do_sympy_compute(plan["op"], plan)
                expl = llm_explain(prompt, result)
                return result + ("\n\n" + expl if expl else "")
            except Exception:
                # fall back to direct LLM answer (reason/explain)
                try:
                    resp = client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role":"system","content":"You are a helpful math tutor. Use LaTeX for math, be concise."},
                            {"role":"user","content":prompt}
                        ],
                        temperature=0.2,
                    )
                    return resp.choices[0].message.content.strip()
                except Exception as e:
                    return f"LLM error: {e}"

    # Last resort
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
st.sidebar.subheader("LLM settings")
use_llm = st.sidebar.checkbox("Use LLM assistance (better parsing + explanations)", value=True)
st.sidebar.markdown("### 🔐 OpenAI status")
st.sidebar.write("✅ Connected" if OPENAI_AVAILABLE else "❌ No API key found")
if use_llm and not OPENAI_AVAILABLE:
    st.sidebar.warning("Set OPENAI_API_KEY via .env (local) or Streamlit Secrets (cloud).")

st.sidebar.markdown("---")
st.sidebar.subheader("Examples")
st.sidebar.code("derivative of sinx", language="text")
st.sidebar.code("integrate e^x", language="text")
st.sidebar.code("limit (1+1/n)^n as n->oo", language="text")
st.sidebar.code("solve {x+y=3, x-y=1}", language="text")
st.sidebar.code("matrix eigenvalues [[1,2],[3,4]]", language="text")

# ---------------- Chat history ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        render_reply(m["content"])

# ---------------- Input & Routing ----------------
user_text = st.chat_input("Type your math question or instruction…")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        render_reply(user_text)

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
        render_reply(reply)
