import asyncio
import inspect
import json
import time
import uuid

from fastapi import BackgroundTasks, FastAPI

from ..examples.prompt_machine import database, get_prompts_for_task_id, setup_database
from .nbasql import get_data_for_question_prompt

promptApiApp = FastAPI(root_path="/prompt")
promptApiApp.jobhistory = {}


@promptApiApp.on_event("startup")
async def database_connect():
    await database.connect()
    await setup_database(database)


@promptApiApp.on_event("shutdown")
async def database_disconnect():
    await database.disconnect()


def sleepy_prompt(input):
    # parse input into int
    input = int(input)
    # sleep for input seconds
    time.sleep(input)
    return "done sleeping for {} seconds".format(input)


class CallWithId:
    def __init__(self, func, task_id):
        self.func = func
        self.task_id = task_id

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def text_to_sparql(text):
    result = asyncio.run(get_data_for_question_prompt(text))
    return {"answer": result}


prompt_lookup = {
    "echo": lambda x: x,
    "sleep": sleepy_prompt,
    "text2sparql": text_to_sparql,
}


def process_prompt(task_id: str):
    try:
        promptApiApp.jobhistory[task_id]["status"] = "received"
        prompt_name = promptApiApp.jobhistory[task_id].get("prompt_name")
        input = promptApiApp.jobhistory[task_id].get("input")
        prompt = prompt_lookup.get(prompt_name)
        if prompt:
            promptApiApp.jobhistory[task_id]["status"] = "processing"
            promptApiApp.jobhistory[task_id]["result"] = CallWithId(prompt, task_id)(
                input
            )
            promptApiApp.jobhistory[task_id]["status"] = "complete"
        else:
            promptApiApp.jobhistory[task_id]["status"] = "error"
            promptApiApp.jobhistory[task_id][
                "error"
            ] = f"Prompt [{prompt_name}] not found"
    except Exception as e:
        promptApiApp.jobhistory[task_id]["error"] = str(e)
        promptApiApp.jobhistory[task_id]["status"] = "error"


@promptApiApp.get("/")
async def read_root(prompt_name: str, input: str, background_tasks: BackgroundTasks):
    taskuid = str(uuid.uuid4())
    promptApiApp.jobhistory[taskuid] = {}
    promptApiApp.jobhistory[taskuid]["status"] = "created"
    promptApiApp.jobhistory[taskuid]["prompt_name"] = prompt_name
    promptApiApp.jobhistory[taskuid]["input"] = input
    background_tasks.add_task(process_prompt, taskuid)
    return {"jobid": taskuid}


@promptApiApp.get("/status/{uid}")
async def read_status(uid: str):
    hopeful = await get_prompts_for_task_id(uid)
    return {"status": str(hopeful)}
