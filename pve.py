#!/usr/bin/env python3

import argparse
import asyncio
import logging
import os
import re
from sqlite3 import Row
from typing import Optional

import aiosqlite
import openai
import tiktoken
from aiolimiter import AsyncLimiter
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

# Initialize the OpenAI client with the API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_API_ORG")
if not openai.api_key:
    raise ValueError(
        "The environment variable OPENAI_API_KEY is not set. Please check your"
        " environment file."
    )
if not openai.organization:
    raise ValueError(
        "The environment variable OPENAI_API_ORG is not set. Please check your"
        " environment file."
    )
CLIENT = openai.AsyncOpenAI()

# OpenAI request parameters
MODEL = "gpt-4o-mini"
ENCODING = tiktoken.encoding_for_model(MODEL)
TOKEN_LIMITER = AsyncLimiter(1.25e6, 1)
REQUEST_LIMITER = AsyncLimiter(250, 1)
CONNECTION_LIMITER = AsyncLimiter(2000)

################################################################################


class PValueObject(BaseModel):
    p: str = Field(pattern=r"^<0\.\d+|[01]?\.\d+$")
    statistic: Optional[str]
    quotation: str


################################################################################


async def chat(system_message, prompt, text):
    # Format the prompt
    prompt = prompt.format(chunk=text)

    # Acquire tokens needed for prompt
    n_tokens = len(ENCODING.encode(system_message)) + len(ENCODING.encode(prompt))
    await TOKEN_LIMITER.acquire(n_tokens)

    # Acquire request and conenction
    async with REQUEST_LIMITER:
        async with CONNECTION_LIMITER:
            response = await CLIENT.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
            )

    return response.choices[0].message.content


################################################################################


def validate(extracted_data, chunk):
    # Validate and clean the extracted JSON data
    p_values = []
    response_objects = re.findall(
        r"\{.*?\}", extracted_data, flags=re.MULTILINE | re.DOTALL
    )
    for json_obj in response_objects:
        try:
            p_val_obj = PValueObject.model_validate_json(json_obj)
            if verify_quote(
                p_value=p_val_obj.p, quotation=p_val_obj.quotation, chunk=chunk["text"]
            ):
                p_values.append(p_val_obj)
        except Exception as e:
            logging.error(
                "Failed to parse JSON output from the model response: %s, %s",
                e,
                json_obj,
            )

    return p_values


################################################################################


# Checking if there's a quotation in the text
def verify_quote(quotation, p_value, chunk):
    # Eliminate unnecessary whitespace
    quotation = re.sub(r"\s+", " ", quotation)
    p_value = re.sub(r"\s+", " ", p_value)
    chunk = re.sub(r"\s+", " ", chunk)

    # If the p-value isn't in the quotation...
    patt = rf"[^\w\s]*{re.escape(p_value)}[^\w\s]*"
    if not re.search(patt, quotation):
        # ... then the quotation is invalid.
        return False
    # Otherwise, match the quotation to the text.
    # NOTE: Short strings must be handled separately.
    if len(quotation) < 10:
        pattern = re.escape(quotation)
    else:
        escaped_quotation_start = re.escape(quotation[:5])
        escaped_quotation_end = re.escape(quotation[-5:])
        pattern = rf"{escaped_quotation_start}.*?{escaped_quotation_end}"

    # Check whether the quotation is in the chunk
    if re.search(pattern, chunk, flags=re.MULTILINE | re.DOTALL):
        return True
    else:
        return False


################################################################################


async def insert_p_values(p_values, chunk_id, conn):
    async with conn.cursor() as cur:
        # Insert the p_values into the database
        for p_value in p_values:
            await cur.execute(
                """
                    INSERT INTO p_values
                    (chunk_id, quotation, test_statistic, p_value)
                    VALUES
                    (?, ?, ?, ?)
                """,
                (chunk_id, p_value.quotation, p_value.statistic, p_value.p),
            )

        # Mark the chunk as processed
        await cur.execute(
            """
            UPDATE chunks
            SET processed = TRUE
            WHERE ID = ?
            """,
            (chunk_id,),
        )

        await conn.commit()


################################################################################


async def process_chunk(chunk, conn):
    # Use LLM to extract the p_values
    extracted_data = await chat(chunk["system_message"], chunk["prompt"], chunk["text"])

    # extract p_values from the response
    p_values = validate(extracted_data, chunk)

    # insert the p_values into the database
    await insert_p_values(p_values, chunk["id"], conn)


################################################################################


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract p-values and test statistics from scientific texts."
    )
    parser.add_argument("--max_chunks", default=1000)
    args = parser.parse_args()

    # Connect to the database
    conn = await aiosqlite.connect("p.db")
    conn.row_factory = Row

    # Set up logging to `p.log`
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("p.log")],
    )
    logging.info("Starting main processing")

    # Get chunks from database
    async with conn.cursor() as cur:
        await cur.execute(
            """
                SELECT *
                FROM chunks
                WHERE NOT processed
                LIMIT ?
            """,
            (args.max_chunks,),
        )
        chunks = await cur.fetchall()

    # Process chunks in parallel
    tasks = [process_chunk(c, conn) for c in chunks]
    await tqdm.gather(*tasks)

    # Close the database
    await conn.close()


################################################################################

if __name__ == "__main__":
    asyncio.run(main())
