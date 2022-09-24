import asyncio
import json
import logging
import os
import time
from functools import partial
from typing import List

import altair as alt
import arel
import faiss
import numpy as np
import pandas as pd
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..core import Portfolio, SketchPad
from ..metrics import strings_from_sketchpad_sketches
from . import auth, data, models
from .deps import *

# logging.basicConfig()
# logging.getLogger("databases").setLevel(logging.DEBUG)

# ## PLAN FOR DIRECTORY STRUCTURE?

# main.py (very short, just create app method, that registers everything)
# pages.py, or is all the "browse-paths"
# components.py or is a bunch of {api}/name
# api.py or is the actual api
# data.py or is the actual database creation, and raw object models
# aut.py the auth stuff
# settings.py the settings for the app


# NOTE: The client API (parent sketch library), will reference in code, the component API,
#  in order to embed capability in jupyter notebooks or streamlit apps (figure out how to embed)
# the component into the client.
# client library, can make HTML blocks, with appropriate "wrapping" (div context wrapper, some sizing stuff, basic composition)
#  -> and then render those or embed them into the client [[ Important to render with a streamlit example for demo purposes ]]
# https://docs.streamlit.io/library/components/create
# Rendering Python objects having methods that output HTML, such as IPython __repr_html__.

# components have a 'rapid render' mode (use another request lifecycle to get data) or a "synchronous" -> oh,
# async or sync mode.. Async mode includes client code in JS for query of data later.
# offline / online mode could be useful, to give things like "always on" or "self-updating" ones.


dir_path = os.path.dirname(os.path.realpath(__file__))

app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(dir_path, "templates"))

# from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
# app.add_middleware(HTTPSRedirectMiddleware)

externalApiApp = FastAPI(root_path="/api")

externalApiApp.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/api", externalApiApp)
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(dir_path, "static")),
    name="static",
)


@app.on_event("startup")
async def database_connect():
    await database.connect()
    await data.setup_database(database)


@app.on_event("shutdown")
async def database_disconnect():
    await database.disconnect()


@app.on_event("startup")
async def setup_index():
    global app
    try:
        index = faiss.read_index(os.path.join(settings.faiss_path, "trained.index"))
    except:
        index = None
    app.index = index
    if settings.setup_fake_users:
        if await data.count_users(database) == 0:
            await data.add_user(
                database,
                "justin",
                "Justin Waugh",
                "justin@approximatelabs.com",
                "$2b$12$xbAAbEMA8UY8iloOrVTxmuDHPbwmpX0TzLNtjHr8BCmFIZvSMVc2.",
            )
            await data.add_user(
                database,
                "mike",
                "James?",
                "mike@approximatelabs.com",
                "$2b$12$xbAAbEMA8UY8iloOrVTxmuDHPbwmpX0TzLNtjHr8BCmFIZvSMVc2.",
            )
            await data.add_user(
                database,
                "datAI",
                "the AI",
                "ai@approximatelabs.com",
                "$2b$12$xbAAbEMA8UY8iloOrVTxmuDHPbwmpX0TzLNtjHr8BCmFIZvSMVc2.",
            )
            await data.add_apikey(
                database,
                "justin",
                "8f79be2b6d0d47ccb8192e46f38c80ce",
                "hello",
                "2025-01-01T00:00:00Z",
            )
            await data.add_apikey(
                database,
                "datAI",
                "5a3556bb2de44a73ab2e5643cb633a6c",
                "hello",
                "2025-01-01T00:00:00Z",
            )


# Set up data (for now just storing stuff on app...)
app.portfolio = Portfolio()

if _debug := settings.debug:
    hot_reload = arel.HotReload(paths=[arel.Path(dir_path)])
    app.add_websocket_route("/hot-reload", route=hot_reload, name="hot-reload")
    app.add_event_handler("startup", hot_reload.startup)
    app.add_event_handler("shutdown", hot_reload.shutdown)
    templates.env.globals["DEBUG"] = _debug
    templates.env.globals["hot_reload"] = hot_reload
    print(("=" * 30 + "\n") * 2 + " " * 12 + "DEBUG\n" + ("=" * 30 + "\n") * 2)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    request.app.database = database
    return response


@app.get("/login")
async def login(request: Request, redirect_uri: str = "/"):
    return templates.TemplateResponse(
        "login.html", {"request": request, "redirect_uri": redirect_uri}
    )


@app.get("/logout")
async def logout(redirectResponse=Depends(auth.logout)):
    return redirectResponse


# Any 401 sends to login page
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        return RedirectResponse(
            app.url_path_for(name="login")
            + ("" if request.url.path == "/" else f"?redirect_uri={request.url.path}")
        )
    elif exc.status_code == 404:
        return templates.TemplateResponse("page/404.html", {"request": request})
    raise exc


@app.get("/404")
async def not_found(request: Request):
    return templates.TemplateResponse("page/404.html", {"request": request})


@app.get("/references")
async def references(
    request: Request, user: auth.User = Depends(auth.get_browser_user)
):
    pf = await data.get_portfolio(database, user.username)
    return templates.TemplateResponse(
        "page/references.html",
        {
            "request": request,
            "user": user,
            "portfolio": pf,
        },
    )


@app.get("/chat")
async def chat(request: Request, user: auth.User = Depends(auth.get_browser_user)):
    return templates.TemplateResponse(
        "page/chat.html",
        {
            "request": request,
            "user": user,
        },
    )


@externalApiApp.get("/get_thread_ids")
async def get_thread_ids(
    request: Request, user: auth.User = Depends(auth.get_token_user)
):
    return await data.get_thread_ids(database)


@app.get("/history")
async def history(request: Request, user: auth.User = Depends(auth.get_browser_user)):
    return templates.TemplateResponse(
        "page/promptHistory.html",
        {
            "request": request,
            "user": user,
        },
    )


@app.get("/promptHistory")
async def getPromptHistory(
    request: Request, user: auth.User = Depends(auth.get_browser_user)
):
    return [
        {
            "id": 17,
            "time": "2022-09-23T16:06:42+00:00",
            "inputs": [
                {"name": "apple", "value": "Apples are very warm when baked."},
                {"name": "innerMonologue", "value": "I'm hungry"},
            ],
            "output": "Mmm, food tastes good.",
        },
        {
            "id": 18,
            "time": "2022-09-23T16:06:55+00:00",
            "inputs": [
                {"name": "apple", "value": "Apples are very cold when frozen."},
                {"name": "innerMonologue", "value": "I'm hungry"},
            ],
            "output": "Crunchy frozen apples, come and get 'em!",
        },
    ]


@app.websocket("/ws/chat")
async def chat_socket(
    websocket: WebSocket,
    user: auth.User = Depends(auth.get_websocket_user),
    thread_id: str = "default",
):
    if user.username:
        await manager.connect(thread_id, websocket, user.username)
        # gather all responses
        messages = await data.get_messages(database, thread_id)
        for messagedata, datetime in messages:
            messagedata.update({"datetime": datetime, "replay": True})
            await websocket.send_json(messagedata)
        await manager.broadcast(
            thread_id, {"message": "joined", "sender": user.username, "meta": True}
        )
        try:
            while True:
                wsdata = await websocket.receive_json()
                if wsdata["message"] == "sudo clear thread":
                    await data.clear_thread(database, thread_id)
                    continue
                wsdata.update({"sender": user.username})
                await data.add_message(database, thread_id, wsdata, user.username)
                await manager.broadcast(thread_id, wsdata)
        except WebSocketDisconnect:
            manager.disconnect(thread_id, websocket, user.username)
            await manager.broadcast(
                thread_id, {"message": "left", "sender": user.username, "meta": True}
            )


@app.get("/reference/{reference_id}")
async def reference(
    request: Request,
    reference_id: str,
    user: auth.User = Depends(auth.get_browser_user),
):
    pf = await data.get_reference_portfolio(database, reference_id, user.username)
    return templates.TemplateResponse(
        "page/reference.html",
        {
            "request": request,
            "user": user,
            "portfolio": pf,
            "reference_id": reference_id,
        },
    )


@app.get("/sketchpad/{sketchpad_id}")
async def sketchpad(
    request: Request,
    sketchpad_id: str,
    user: auth.User = Depends(auth.get_browser_user),
):
    sketchpad = await data.get_sketchpad(database, sketchpad_id, user.username)
    return templates.TemplateResponse(
        "page/sketchpad.html",
        {
            "request": request,
            "user": user,
            "sketchpad": sketchpad,
            "sketchpad_json": json.dumps(sketchpad.to_dict(), indent=2),
            "sketchpad_id": sketchpad_id,
        },
    )


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


@app.get("/refresh_index")
async def refresh_index(
    request: Request, user: auth.User = Depends(auth.get_browser_user)
):
    import time

    from sentence_transformers import SentenceTransformer

    global local_cache
    local_cache = {}

    model = SentenceTransformer("all-MiniLM-L6-v2")
    # # iterate over references, and add to index
    # short_ids, references = zip(*[x async for x in data.get_references(database)])
    # embeddings = model.encode([r.to_searchable_string() for r in references])

    sketchpad_sentences, short_ids = [], []

    # TODO: replace this with logger
    st = time.time()
    print(f"{time.time()}  Rebuilding index... gathering sketchpads")
    async for sketchpad in data.get_sketchpads(database, user.username):
        sketchpad_sentences.append(
            sketchpad.reference.to_searchable_string()
            + strings_from_sketchpad_sketches(sketchpad)
        )
        short_ids.append(sketchpad.reference.short_id)
        if len(sketchpad_sentences) % 500 == 0:
            print(len(sketchpad_sentences))
    print(f"{time.time()}  Rebuilding index... generating embeddings")
    # do in batches and print out batches
    embeddings = []
    for chunk in chunks(sketchpad_sentences, 500):
        embeddings.append(model.encode(chunk))
        print(f"{time.time()}... {(len(embeddings)+1)*500}")
    embeddings = np.concatenate(embeddings)
    print(f"{time.time()}  Rebuilding index... building index")
    # see https://github.com/facebookresearch/faiss/blob/main/benchs/bench_hnsw.py

    # index = faiss.IndexFlatL2(embeddings.shape[1])
    index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
    index.hnsw.efConstruction = 40
    index2 = faiss.IndexIDMap(index)
    print("adding short_ids as reference")
    index2.add_with_ids(
        embeddings,
        np.array(short_ids, dtype=np.int64),
    )
    global app
    app.index = index2
    print("saving index...")
    faiss.write_index(index2, os.path.join(settings.faiss_path, "trained.index"))
    print("done and saved... took ", time.time() - st)
    return "Okay..."


@app.get("/search")
async def search(
    request: Request, user: auth.User = Depends(auth.get_browser_user), q: str = ""
):

    # obviously no need to normally get the whole thing... but for now, we'll do it.
    if q:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")

        query_vector = model.encode([q])
        index = app.index
        D, I = index.search(query_vector, 5)
        indexes = list(I[0])
        sketchpads = {
            i: x
            async for i, x in data.get_most_recent_sketchpads_by_reference_short_ids(
                database, indexes, user.username
            )
        }
        sketchpads = [sketchpads[i] for i in indexes]
        pf = Portfolio(sketchpads=sketchpads)
    else:
        pf = Portfolio()

    return templates.TemplateResponse(
        "page/search.html",
        {
            "request": request,
            "user": user,
            "portfolio": pf,
            "q": q,
        },
    )


@app.get("/apple")
async def apple(request: Request, user: auth.User = Depends(auth.get_browser_user)):
    return "🍎"


@app.get("/")
async def root(request: Request, user: auth.User = Depends(auth.get_browser_user)):
    if f"root-{user.username}" in local_cache:
        return local_cache[f"root-{user.username}"]
    pf = await data.get_portfolio(database, user.username)
    local_cache[f"root-{user.username}"] = templates.TemplateResponse(
        "page/root.html",
        {
            "request": request,
            "sketchcount": len(pf.sketchpads),
            "user": user,
        },
    )
    return local_cache[f"root-{user.username}"]


@app.post("/token")
async def login_for_access_token(
    token_response=Depends(auth.login_for_access_token),
):
    return token_response


# TODO: rename this spectogram?
@app.get("/cardinality_histogram")
async def cardhisto(request: Request, user: auth.User = Depends(auth.get_browser_user)):
    if f"cardhisto-{user.username}" in local_cache:
        return local_cache[f"cardhisto-{user.username}"]
    pf = await data.get_portfolio(database, user.username)

    cards = np.array(
        [
            x.get_sketchdata_by_name("HyperLogLog").count()
            for x in pf.sketchpads.values()
        ]
    )
    # hist, bins = np.histogram(cards)
    upper = pow(2, np.ceil(np.log(np.max(cards) / np.log(2))))
    bins = np.geomspace(1, upper, num=100)
    hist, *_ = np.histogram(cards, bins=bins)
    # should be geometric mean...
    df = pd.DataFrame({"x": bins[:-1], "x1": bins[1:], "y": hist})
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                "x", title="Unique Count", scale=alt.Scale(type="log", domainMin=1)
            ),
            # x2="x2",
            # y=alt.Y("y", title="Number", scale=alt.Scale(type="log")),
            y=alt.Y("y", title="Count"),
        )
        .properties(width="container", height=300)
    )
    local_cache[f"cardhisto-{user.username}"] = chart.to_dict()
    return local_cache[f"cardhisto-{user.username}"]


@app.get("/cardinality_history")
async def cardinality_history(
    request: Request,
    reference_id: str,
    user: auth.User = Depends(auth.get_browser_user),
):
    pf = await data.get_reference_portfolio(database, reference_id, user.username)
    df = pd.DataFrame(
        {
            "date": [s.metadata["creation_start"] for s in pf.sketchpads.values()],
            "cardinality_estimate": [
                s.get_sketchdata_by_name("HyperLogLog").count()
                for s in pf.sketchpads.values()
            ],
        }
    )
    chart = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(color="blue"))
        .encode(x="date:T", y="cardinality_estimate:Q")
    )
    return chart.to_dict()


@externalApiApp.post("/upload_sketchpad")
# The request to send a sketchpad to our services
async def upload_sketchpad(
    sketchpad: models.SketchPad, user: auth.User = Depends(auth.get_token_user)
):
    global local_cache
    local_cache = {}
    # Ensure sketchpad parses correctly
    SketchPad.from_dict(sketchpad.dict())
    # add the pydantic sketchpad directly to database
    await data.add_sketchpad(database, user.username, sketchpad)
    return {"status": "ok"}


@externalApiApp.post("/upload_portfolio")
async def upload_portfolio(
    sketchpads: List[models.SketchPad], user: auth.User = Depends(auth.get_token_user)
):
    # Ensure sketchpad parses correctly
    for sketchpad in sketchpads:
        SketchPad.from_dict(sketchpad.dict())
    # add the pydantic sketchpad directly to database
    add_sketchpad = partial(data.add_sketchpad, database, user.username)
    await asyncio.gather(*map(add_sketchpad, sketchpads))
    return {"status": "ok"}


@externalApiApp.get("/get_approx_best_joins")
async def get_approx_best_joins(
    sketchpad_ids: List[str], user: auth.User = Depends(auth.get_token_user)
):
    pf = await data.get_portfolio(database, user.username)
    pf = pf.get_approx_pk_sketchpads()
    myspads = [
        x
        async for x in data.get_sketchpads_by_id(database, sketchpad_ids, user.username)
    ]
    possibilities = []
    for sketch in myspads:
        high_overlaps = pf.closest_overlap(sketch)
        for score, sketchpad in high_overlaps:
            possibilities.append(
                {
                    "score": score,
                    "left_sketchpad": sketch.to_dict(),
                    "right_sketchpad": sketchpad.to_dict(),
                }
            )
    return sorted(possibilities, key=lambda x: x["score"], reverse=True)[:5]


@externalApiApp.get("/component/get_approx_best_joins")
async def get_approx_best_joins(
    request: Request, best_joins=Depends(get_approx_best_joins)
):
    pf = Portfolio(
        sketchpads=[SketchPad.from_dict(x["right_sketchpad"]) for x in best_joins]
    )
    return templates.TemplateResponse(
        "component/references.html",
        {"request": request, "portfolio": pf},
    )
