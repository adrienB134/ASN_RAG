"""Main entrypoint for the app."""

import logging
import os
from pathlib import Path
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from lm.lm import gpt_3_5_turbo, gpt_4_turbo, ollama_lm
from rag import RAG
from utils.schemas import ChatResponse

rag = RAG(lm=gpt_3_5_turbo, final_writer=ollama_lm)
app = FastAPI()
templates = Jinja2Templates(directory="templates")

load_dotenv()


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    chat_history = []
    chat_history.append(("", "Bonjour, je suis ChatASN votre assistant de recherche."))
    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.model_dump())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.model_dump())

            model_answer, sources = rag.query(question)

            # chat_history.append((question, result["answer"]))
            result_resp = ChatResponse(
                sender="bot",
                message=model_answer.replace("\n", "<br>"),
                type="stream",
            )
            await websocket.send_json(result_resp.model_dump())

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.model_dump())

            for source in sources:
                sources_resp = ChatResponse(
                    sender="bot",
                    message=source["source"],
                    type="sources",
                )
                await websocket.send_json(sources_resp.model_dump())

        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.model_dump())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9001)
