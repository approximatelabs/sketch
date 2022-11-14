import asyncio
import inspect
import json
import time
import uuid

from fastapi import BackgroundTasks, FastAPI

from ..examples.prompt_machine import database, get_prompts_for_task_id, setup_database
from .nbasql import get_data_for_question_prompt

promptApiApp = FastAPI(root_path="/prompt")


@promptApiApp.on_event("startup")
async def database_connect():
    await database.connect()
    await setup_database(database)


@promptApiApp.on_event("shutdown")
async def database_disconnect():
    await database.disconnect()


class CallWithId:
    def __init__(self, func, task_id):
        self.func = func
        self.task_id = task_id

    def __call__(self, *args, **kwargs):
        result = self.func(*args, **kwargs)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        return result


prompt_lookup = {
    "text2sparql": get_data_for_question_prompt,
}


@promptApiApp.get("/")
async def read_root(prompt_name: str, input: str, background_tasks: BackgroundTasks):
    taskuid = str(uuid.uuid4())
    background_tasks.add_task(
        CallWithId(prompt_lookup.get(prompt_name), taskuid), input
    )
    return {"jobid": taskuid}


@promptApiApp.get("/status/{uid}")
async def read_status(uid: str):
    return await get_prompts_for_task_id(uid)
