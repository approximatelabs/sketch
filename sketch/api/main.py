import os
import time

import altair as alt
import arel
import numpy as np
import pandas as pd
from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseSettings

from ..core import Portfolio, SketchPad
from . import auth, models


# https://fastapi.tiangolo.com/advanced/settings/
class Settings(BaseSettings):
    app_name: str = "SketchAPI"
    sqlitepath: str = "sqlite+aiosqlite:///test.db"
    debug: bool = False


dir_path = os.path.dirname(os.path.realpath(__file__))

settings = Settings()
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

# Set up data (for now just storing stuff on app...)
app.portfolio = Portfolio()

if _debug := settings.debug:
    hot_reload = arel.HotReload(paths=[arel.Path(".")])
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
    raise exc


@app.get("/")
async def root(
    request: Request, user: auth.User = Depends(auth.get_current_active_user)
):
    return templates.TemplateResponse(
        "root.html",
        {"request": request, "sketchcount": len(app.portfolio.sketchpads)},
    )


@app.get("/apple")
async def apple(
    request: Request, user: auth.User = Depends(auth.get_current_active_user)
):
    return "🍎"


@app.post("/token")
async def login_for_access_token(
    token_response=Depends(auth.login_for_access_token),
):
    return token_response


@app.get("/cardinality_histogram")
async def cardhisto(request: Request):
    cards = np.array(
        [
            x.get_sketchdata_by_name("HyperLogLog").count()
            for x in app.portfolio.sketchpads.values()
        ]
    )
    # hist, bins = np.histogram(cards)
    upper = pow(2, np.ceil(np.log(np.max(cards) / np.log(2))))
    bins = np.geomspace(1, upper, num=100)
    hist, *_ = np.histogram(cards, bins=bins)
    # should be geometric mean...
    data = pd.DataFrame({"x": bins[:-1], "x1": bins[1:], "y": hist})
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("x", title="Unique Count", scale=alt.Scale(type="log")),
            x2="x2",
            # y=alt.Y("y", title="Number", scale=alt.Scale(type="log")),
            y=alt.Y("y", title="Count"),
        )
        .properties(width="container", height=300)
    )
    return chart.to_dict()


@externalApiApp.post("/")
# The request to send a sketchpad to our services
async def api(sketchpad: models.SketchPad, token: str | None = Cookie(default=None)):
    sketchpad_dict = sketchpad.dict()
    sp = SketchPad.from_dict(sketchpad_dict)
    app.portfolio.add_sketchpad(sp)
