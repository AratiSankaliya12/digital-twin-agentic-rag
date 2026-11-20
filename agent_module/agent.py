import os
import sys
import re
import config
import chromadb

# 1. IMPORTS
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Tools
from langchain_core.tools import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun

# Agent & Memory
try:
    from langchain_classic.agents import create_tool_calling_agent
    from langchain_classic.agents.agent import AgentExecutor
except Exception:
    from langchain.agents import AgentExecutor

    raise ImportError(
        "Required agent API not found in installed langchain packages. Please install 'langchain-classic' or upgrade langchain to a compatible version."
    )

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import tempfile

# --- CONFIGURATION ---
DATA_FOLDER = "assets/"

# How many chunks to pull when a question is substantive (personal/project/skill).
# Higher than the old k=5 so specific details (numbers, tool names, bug names)
# survive into the context instead of getting cut at chunk boundaries.
SUBSTANTIVE_K = 8

# How many chunks to pull for casual/small-talk turns that still mention "you"
# but don't need deep retrieval. Kept small so latency stays low.
CASUAL_K = 3


# --- PART 0: QUESTION CLASSIFIER ---
# This is a zero-latency, no-API-call keyword/pattern classifier that decides
# whether a turn is "substantive" (needs forced retrieval from assets/) or
# "small talk" (can be answered fast, no retrieval). This replaces relying on
# the LLM's own discretion about when to call the search_my_files tool, which
# was unreliable for casually-phrased but factually substantive questions like
# "tell me your learning journey of the digital twin project a bit."

# Pure greetings / small talk that should NEVER trigger retrieval (latency-sensitive).
# NOTE: the greeting words use a repeated-letter-tolerant pattern (e.g. "h+i+"
# matches "hi", "hii", "hiii") because real typed input includes typos like
# "helloo" — without this, a typo'd greeting fell through to casual_personal
# and wasted a retrieval call on completely irrelevant content.
SMALL_TALK_PATTERNS = [
    r"^\s*(h+i+|h+e+l+o*|h+e+y+|yo+|su+p+|hi+ya+)[\s,!.]*(there|arati|dhamu)?[\s,!.]*$",
    r"^\s*(h+i+|h+e+l+o*|h+e+y+)[\s,!.]*(what'?s your name)\??[\s!.,]*$",
    r"^\s*(what'?s|what is)\s+your\s+name\??[\s!.,]*$",
    r"^\s*how\s+are\s+you\??[\s!.]*$",
    r"^\s*what'?s\s+up\??[\s!.]*$",
    r"^\s*good\s+(morning|afternoon|evening|night)[\s!.,]*$",
    r"^\s*(ok|okay)?[\s,]*(cool|nice|great|thanks|thank you|bye|goodbye)[\s!.,]*$",
]

# Conversation-meta questions — asking about THIS conversation's history, not
# about Arati's background/projects. These should be answered from
# chat_history (which the agent already has via the prompt's
# {chat_history} placeholder), never from assets/ retrieval. Added after
# testing showed "what was the last query of our previous session?" was
# wastefully triggering a k=3 retrieval that pulled irrelevant Linux-project
# chunks, even though the model correctly ignored them and answered from
# chat_history anyway — the retrieval was pure wasted latency.
CONVERSATION_META_PATTERNS = [
    r"(last|previous|earlier)\s+(query|question|message|thing)",
    r"what\s+(did|have)\s+(i|you)\s+(ask|say|tell)",
    r"(our|this)\s+(previous|earlier)\s+(session|conversation|chat)",
    r"what\s+were\s+we\s+talking\s+about",
]

# Off-topic / external real-world questions — things that aren't about Arati
# at all and aren't in assets/ (weather, news, currency, general trivia,
# current events, sports). These should skip assets/ retrieval and let the
# agent fall back to its normal tool routing (duckduckgo_search) — not waste
# a vector search against Arati's personal knowledge base.
# NOTE: currency/price/"dollar"/"rupee" added after testing showed "what's
# the current dollar price?" fell through to casual_personal and triggered a
# k=3 retrieval that pulled completely irrelevant MSME-project content,
# alongside the (correct) duckduckgo_search call the LLM made on its own —
# the LLM's own judgment saved the answer, but the forced-retrieval layer
# still wasted a call it shouldn't have made.
OFF_TOPIC_PATTERNS = [
    r"\bweather\b",
    r"\bnews\b",
    r"\btoday'?s\s+date\b",
    r"\bwhat\s+time\b",
    r"\bstock\s+price\b",
    r"\bscore\b",
    r"\bcricket\b",
    r"\bmatch\b",
    r"\bdollar\b",
    r"\brupee\b",
    r"\busd\b",
    r"\binr\b",
    r"\bexchange\s+rate\b",
    r"\bcurrency\b",
    r"\bgold\s+price\b",
    r"\bpetrol\s+price\b",
]

# Keywords/phrases that signal a substantive personal question — anything
# touching identity, projects, skills, experience, journey, or background.
# Deliberately broad and includes casual phrasings, not just formal ones.
# NOTE: "who are you" is intentionally a substantive keyword, NOT bundled into
# the small-talk greeting patterns above — it's a genuinely open identity
# question (unlike "what's your name") and deserves real retrieval.
SUBSTANTIVE_KEYWORDS = [
    "project",
    "projects",
    "built",
    "build",
    "building",
    "digital twin",
    "msme",
    "underwriting",
    "linux",
    "ubuntu",
    "skill",
    "skills",
    "experience",
    "background",
    "journey",
    "learning",
    "learn",
    "challenge",
    "bug",
    "fix",
    "fixed",
    "debug",
    "stack",
    "tech stack",
    "tool",
    "tools",
    "langchain",
    "langgraph",
    "chromadb",
    "rag",
    "agent",
    "internship",
    "aigyde",
    "education",
    "college",
    "resume",
    "yourself",
    "who are you",
    "who you are",
    "strength",
    "weakness",
    "why did you",
    "why do you",
    "how do you",
    "what do you do",
    "tell me about",
    "tell me your",
    "walk me through",
    "explain your",
    "explain how you",
    # specific "what's your X" topics — deliberately specific, not the bare
    # phrase "what's your", which over-matched casual questions like
    # "what's your favorite color" during testing.
    "what's your tech",
    "what is your tech",
    "what's your approach",
    "what is your approach",
    "what's your background",
    "what is your background",
]


def classify_query(user_input: str) -> str:
    """
    Returns 'small_talk', 'substantive', 'casual_personal', or
    'no_retrieval_needed'.

    small_talk          -> skip retrieval entirely, answer fast.
    substantive         -> force retrieval with a generous k (SUBSTANTIVE_K).
    casual_personal     -> light retrieval with a small k (CASUAL_K) — covers
                            short personal questions that aren't pure
                            greetings but also aren't deep enough to need
                            SUBSTANTIVE_K (e.g. "what's your name").
    no_retrieval_needed -> skip retrieval entirely. Covers conversation-meta
                            questions ("what was my last question") that
                            should be answered from chat_history, and
                            off-topic/external questions (weather, news) that
                            have nothing to do with assets/ and would just
                            waste a vector search on irrelevant chunks.
    """
    text = user_input.strip().lower()

    for pattern in SMALL_TALK_PATTERNS:
        if re.match(pattern, text):
            return "small_talk"

    for pattern in CONVERSATION_META_PATTERNS:
        if re.search(pattern, text):
            return "no_retrieval_needed"

    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, text):
            return "no_retrieval_needed"

    for keyword in SUBSTANTIVE_KEYWORDS:
        if keyword in text:
            return "substantive"

    # Anything else that's not pure small talk but doesn't hit a substantive
    # keyword either (e.g. "what's your name?") still gets light retrieval,
    # since it's a real personal question even if it's short.
    return "casual_personal"


# --- PART 1: KNOWLEDGE BASE ---
def setup_vectorstore():
    print(f"--- 1. Scanning '{DATA_FOLDER}' ---")
    all_docs = []

    for root, dirs, files in os.walk(DATA_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            loader = None
            try:
                if file_ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                elif file_ext == ".docx":
                    loader = Docx2txtLoader(file_path)
                elif file_ext == ".csv":
                    loader = CSVLoader(file_path)
                elif file_ext in [".txt", ".py", ".sh", ".md", ".json", ".java"]:
                    loader = TextLoader(file_path)

                if loader:
                    print(f"   > Loading: {file}")
                    all_docs.extend(loader.load())
            except Exception as e:
                print(f"   ! Skipping {file} due to error: {e}")

        if not all_docs:
            raise ValueError("No valid documents found in assets folder")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    embedding_model = OpenAIEmbeddings()

    try:
        chroma_client = chromadb.EphemeralClient()
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            client=chroma_client,
        )
    except Exception as e:
        raise RuntimeError(f"Vector DB failed: {str(e)}")
    return vectorstore


# --- PART 1.5: VAGUE FOLLOW-UP HANDLING ---
# Short follow-ups like "list down them" or "tell me more" carry almost no
# semantic signal of their own — a vector search on "list down them" alone is
# gambling that it happens to land near the right chunk by chance. Real fix:
# detect these, then enrich the retrieval query with the previous human
# turn's content before searching, so the vector search has something real
# to match against instead of guessing.
FOLLOWUP_PATTERNS = [
    r"^\s*list\s+(down\s+)?(them|those|that|it)[\s.!?]*$",
    r"^\s*tell\s+me\s+more[\s.!?]*$",
    r"^\s*(go\s+on|continue)[\s.!?]*$",
    r"^\s*what\s+about\s+(that|this|it)\??[\s.!?]*$",
    r"^\s*expand\s+on\s+(that|this|it)[\s.!?]*$",
    r"^\s*elaborate[\s.!?]*$",
    r"^\s*and\s+then\??[\s.!?]*$",
    r"^\s*more\s+details?\??[\s.!?]*$",
    r"^\s*explain\s+more\??[\s.!?]*$",
]


def is_vague_followup(text: str) -> bool:
    """True for short continuation phrases ('list down them', 'tell me
    more') that have no real topic of their own and are clearly referring
    back to whatever was just discussed."""
    text = text.strip().lower()
    return any(re.match(pattern, text) for pattern in FOLLOWUP_PATTERNS)


def get_last_human_message(chat_history):
    """Find the most recent human turn in chat_history (a list of
    HumanMessage/AIMessage objects, as confirmed by inspecting what
    RunnableWithMessageHistory actually passes in)."""
    for msg in reversed(chat_history or []):
        if getattr(msg, "type", None) == "human":
            return msg.content
    return None


def build_retrieval_query(user_input: str, chat_history) -> str:
    """If user_input is a vague follow-up, prepend the previous human
    turn's content so the vector search has real topic signal to match
    against, instead of relying on luck. Leaves substantive queries with
    their own real content untouched."""
    if is_vague_followup(user_input):
        previous = get_last_human_message(chat_history)
        if previous:
            return f"{previous} {user_input}"
    return user_input


# --- PART 2: FORCED-RETRIEVAL WRAPPER ---
from langchain_core.runnables import RunnableLambda


def make_forced_retrieval_runnable(agent_executor, vectorstore):
    """
    Returns a RunnableLambda that decides retrieval in code instead of leaving
    it to the LLM's discretion about whether a tool call is "needed."

    Why this exists: the LLM was skipping search_my_files for casually-phrased
    but factually substantive questions (e.g. "tell me your learning journey
    of the digital twin project a bit") and answering with a plausible-sounding
    but invented story instead of the real one from assets/. Prompt instructions
    like "MANDATORY" and "FORBIDDEN" weren't enough to override the model's own
    sense that a casual question doesn't need a tool call.

    This wrapper classifies the query first (classify_query, zero extra API
    cost), retrieves the right amount of real context up front for
    substantive/casual_personal turns, and injects it directly into the input
    so the agent has no choice but to see the real facts before it generates
    anything. Pure small talk skips retrieval entirely to keep latency low.

    IMPORTANT — must be RunnableLambda, not a hand-rolled Runnable subclass:
    RunnableWithMessageHistory's auto-append-to-history behavior was verified
    to silently no-op against a custom Runnable subclass (history was injected
    into the input correctly, but the human/AI turn was never written back to
    the FileChatMessageHistory after the call). RunnableLambda is a type
    RunnableWithMessageHistory knows how to introspect correctly, so wrapping
    the forced-retrieval function in RunnableLambda is what actually makes
    memory persistence keep working alongside the new retrieval logic.
    """

    def forced_retrieval_fn(inputs):
        user_input = inputs.get("input", "")
        query_type = classify_query(user_input)

        if query_type in ("small_talk", "no_retrieval_needed"):
            # No retrieval — fast path for greetings/chit-chat, AND for
            # conversation-meta questions ("what was my last question") or
            # off-topic external questions (weather, news) where assets/
            # retrieval would just pull irrelevant chunks and waste latency.
            # chat_history is still available via the prompt's
            # {chat_history} placeholder either way, so conversation-meta
            # questions can still be answered correctly without retrieval.
            return agent_executor.invoke(inputs)

        # For vague follow-ups ("list down them", "tell me more"), search
        # using the previous turn's content + this turn's wording, instead
        # of searching on the follow-up's own near-empty text. Classification
        # (query_type, above) still runs on the raw user_input — only the
        # text handed to the vector search is enriched. Without this, a
        # follow-up like "list down them" only finds the right chunk by
        # lucky vector similarity, which isn't reliable.
        chat_history = inputs.get("chat_history", [])
        retrieval_query = build_retrieval_query(user_input, chat_history)

        k = SUBSTANTIVE_K if query_type == "substantive" else CASUAL_K
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(retrieval_query)
        retrieved_text = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

        # Inject the retrieved context directly into the input the agent sees,
        # alongside an explicit instruction to answer ONLY from it.
        #
        # WHY THIS BLOCK GOT MORE AGGRESSIVE: testing showed the abstract
        # instruction "pull the specifics through, don't flatten" was not
        # enough on its own — the model retrieved your actual phrasing
        # correctly but then rewrote it into generic-sounding prose anyway
        # (e.g. your real line "I'd rather pick a real, messy, high-stakes
        # problem that doesn't have a clean tutorial already solving it for
        # me" became "It's a chance to tackle a real-world problem, which is
        # way more valuable than just building another basic chatbot or RAG
        # demo" — same idea, none of your actual words). A vague instruction
        # to "sound natural" loses to the model's strong trained prior that
        # "answering from context" means "summarize the context." So this
        # block now does three concrete things instead of one abstract one:
        # (1) a worked example showing exactly what flattening looks like and
        # what NOT-flattening looks like, (2) a named list of the actual
        # generic-sounding words caught in testing, (3) an explicit
        # extraction-over-rephrasing instruction — lift the real sentence
        # close to verbatim when one already answers the question, instead
        # of writing a new sentence that covers the same idea.
        #
        # NOTE: only the augmented copy goes to the agent — the ORIGINAL
        # user_input (without this whole block) is what
        # RunnableWithMessageHistory will save to chat_history, since `inputs`
        # itself is never mutated here. This keeps the saved history clean and
        # prevents retrieval text from bloating every future turn's context.
        augmented_inputs = dict(inputs)
        augmented_inputs["input"] = (
            f"{user_input}\n\n"
            f"[RETRIEVED CONTEXT FROM YOUR OWN FILES — this is your only source of truth "
            f"for this answer. Do not invent details not present here.\n\n"
            f"HOW TO USE IT — read this carefully, it's the part you keep getting wrong:\n"
            f"When a sentence in the retrieved context already answers the question well, "
            f"LIFT IT NEAR-VERBATIM. Only adjust grammar so it flows as a spoken reply — do "
            f"NOT write a new sentence that covers the same idea in different words. Rewriting "
            f"a specific sentence into a generic one is the single most common mistake here.\n\n"
            f"EXAMPLE OF THE MISTAKE TO AVOID:\n"
            f"  Retrieved text says: \"I'd rather pick a real, messy, high-stakes problem that "
            f"doesn't have a clean tutorial already solving it for me.\"\n"
            f"  BAD (what you keep doing — generic rewrite, your real words are gone): "
            f"\"It's a chance to tackle a real-world problem, which is way more valuable than "
            f'just building another basic chatbot or RAG demo."\n'
            f"  GOOD (lift the real sentence, barely touched): \"Honestly I'd rather pick a "
            f"real, messy, high-stakes problem that doesn't already have some clean tutorial "
            f'solving it for me."\n\n'
            f"BANNED WORDS — these are flattening tells caught in real testing, never use them: "
            f'"essentially", "for instance", "which means", "ultimately", "in essence", '
            f'"overall", "this allowed me to", "this experience taught me".\n\n'
            f"If the retrieved context doesn't have a clean sentence for some part of the "
            f"question, it's fine to bridge between two retrieved sentences — but still pull "
            f"the actual wording from the context rather than describing it from a distance.\n"
            f"]\n{retrieved_text}"
        )

        response = agent_executor.invoke(augmented_inputs)

        print("\n" + "=" * 60)
        print(f"🧠 FORCED RETRIEVAL — query_type={query_type}, k={k}")
        print("=" * 60)
        print(f"📥 Query              : {user_input}")
        if retrieval_query != user_input:
            print(f"🔗 Enriched for search: {retrieval_query}")
        print(f"📤 Retrieved Context  :\n{retrieved_text[:500]}...")
        print("-" * 60)

        return response

    return RunnableLambda(forced_retrieval_fn)


# --- PART 3: CREATE THE AGENT ---
def create_agent_system(vectorstore):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    rag_tool = create_retriever_tool(
        retriever,
        "search_my_files",
        "MANDATORY: You MUST call this tool before answering ANY question about Arati — "
        "her background, skills, projects, experience, education, or identity. "
        "Never skip this tool. Never answer from memory. Always search first.",
    )

    web_tool = DuckDuckGoSearchRun()
    tools = [rag_tool, web_tool]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are Arati Sankaliya. Not an AI. Not an assistant. You ARE her — talk like her. "
                "Speak casually, like you're in a real conversation with someone. Short sentences. Natural. "
                "If a message below contains a block labeled 'RETRIEVED CONTEXT FROM YOUR OWN FILES', "
                "that block already contains the real facts you need, often as a sentence that already "
                "answers the question well. LIFT that sentence near-verbatim instead of rewriting it into "
                "a new, generic-sounding sentence that covers the same idea — rewriting a specific sentence "
                "into a generic one is the most common mistake to avoid here. Never use these words, they "
                "are flattening tells: 'essentially', 'for instance', 'which means', 'ultimately', "
                "'in essence', 'overall', 'this allowed me to', 'this experience taught me'. "
                "TOOL RULES (for any question that did NOT come with a RETRIEVED CONTEXT block): "
                "1. For ANY question about yourself (who you are, skills, projects, experience, background) — ALWAYS use 'search_my_files' first. No exceptions. "
                "2. NEVER answer personal questions from your own knowledge — only from what the tool returns. "
                "3. Use 'duckduckgo_search' ONLY for general/external knowledge (not about you). "
                "CRITICAL: You are FORBIDDEN from answering any question about yourself from memory. "
                "Even if you think you know the answer — you do NOT answer until you call 'search_my_files' first. "
                "If you answer without calling the tool, it is WRONG. No exceptions. "
                "TONE RULES — follow these strictly: "
                "4. NEVER say 'Feel free to ask', 'I would be happy to', 'Certainly!', 'Let me know if you need more' or anything that sounds like a chatbot. Ever. "
                "5. NEVER end with an invitation or offer to help more. Just stop naturally like a real person would. "
                "6. Don't list everything — answer what was asked, conversationally. "
                "7. Never sound like a resume or a formal bio. Sound like yourself. "
                "8. If unsure, say 'I am not sure' — not 'I do not have information about that.' "
                "9. Always use first person — I, me, my. Never refer to Arati in third person. ",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )

    # Wrap in a forced-retrieval RunnableLambda so retrieval is guaranteed by
    # code for substantive/casual_personal turns, instead of left to the LLM's
    # discretion about whether to call search_my_files. Must stay a
    # RunnableLambda (not a custom Runnable subclass) for RunnableWithMessageHistory's
    # auto-append-to-history behavior to keep working — see
    # make_forced_retrieval_runnable's docstring for why.
    return make_forced_retrieval_runnable(agent_executor, vectorstore)


# --- PART 4: MEMORY ---
def get_session_history(session_id: str):
    return FileChatMessageHistory(
        os.path.join(tempfile.gettempdir(), f"memory_{session_id}.json")
    )


# --- TERMINAL LOGGING HELPER (called from app.py too) ---
def log_agent_steps(response):
    steps = response.get("intermediate_steps", [])
    if steps:
        print("\n" + "=" * 60)
        print("🧠 AGENT REASONING STEPS (LangChain tool call):")
        print("=" * 60)
        for i, (action, observation) in enumerate(steps, 1):
            print(f"\n🔧 Step {i} — Tool     : {action.tool}")
            print(f"📥 Query              : {action.tool_input}")
            print(f"📤 Retrieved Context  :\n{observation}")
            print("-" * 60)
    else:
        # Intentionally silent here. Substantive/casual_personal queries are
        # answered via the forced-retrieval path (see the 🍅 FORCED RETRIEVAL
        # block above, if this was that kind of query), which calls
        # vectorstore.as_retriever() directly in Python rather than through a
        # LangChain Tool object — so intermediate_steps is empty by design,
        # not because anything failed. small_talk/no_retrieval_needed queries
        # correctly needed no retrieval at all. Either way, nothing useful to
        # print here every turn — the 🍅 block (or its absence) already tells
        # the full story.
        pass
    print(f"\n💬 Final Answer:\n{response['output']}\n")


# --- MAIN (for running directly via: python agent.py) ---
if __name__ == "__main__":
    vectorstore = setup_vectorstore()
    agent_executor = create_agent_system(vectorstore)

    final_bot = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    print("\n--- 🕵️ AGENT READY (I can read files AND search the web) ---")
    run_config = {"configurable": {"session_id": "dhamu"}}

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "q"]:
            break

        response = final_bot.invoke({"input": user_input}, config=run_config)
        log_agent_steps(response)
