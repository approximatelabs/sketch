import json
import os
import time

import altair as alt
import arel
import numpy as np
import pandas as pd
from fastapi import Cookie, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseSettings

from ..core import Portfolio, SketchPad
from . import models


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


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(
        "root.html", {"request": request, "sketchcount": len(app.portfolio.sketchpads)}
    )


@app.get("/cardinality_histogram")
async def cardhisto(request: Request):
    cards = np.array(
        [
            x.get_sketchdata_by_name("HyperLogLog").count()
            for x in app.portfolio.sketchpads.values()
        ]
    )
    hist, bins = np.histogram(cards)
    bin_centers = (bins[0:-1] + bins[1:]) / 2
    # upper = pow(2, np.ceil(np.log(np.max(cards)/np.log(2))))
    # bins = np.geomspace(1, upper, num=20)
    # hist, = np.histogram(cards, bins=bins)
    chart = (
        alt.Chart(pd.DataFrame({"x": bin_centers, "y": hist}))
        .mark_bar()
        .encode(x="x", y="y")
        .properties(width="container", height=300)
    )
    return chart.to_dict()


@externalApiApp.post("/")
# The request to send a sketchpad to our services
async def api(sketchpad: models.SketchPad, token: str | None = Cookie(default=None)):
    sketchpad_dict = sketchpad.dict()
    sp = SketchPad.from_dict(sketchpad_dict)
    app.portfolio.add_sketchpad(sp)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    print(request.cookies)
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
