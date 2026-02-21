## PDF
- our starting point
- probably cloud storage

## PARSER
- extracts text and tables from PDF (parsing only — chunking/embedding happen downstream)
- two options
    - PyMuPDF: text only
        - local
        - free
        - faster, but parallelism is???
    - Aryn: advanced parsing features especially for tabular/unstructured data
        - remote
        - not free ($2 per 1000 pages)
        - API allows for parallel processing
- structure caching: saves information about the page structure for every document
    - right now mostly used to select pages that are structurally similar to the target page to improve LLM training data
    - slightly worried that this might be biasing the model *against* similar chunks
    - worth testing: random distractors vs structurally-similar distractors

## CHUNKING
- splits parsed text into chunks for embedding
- chunk size: currently 4000 chars (in embed.py _chunk_text)
    - bigger chunks = more context per chunk but fewer fit in the LLM window
- overlap: currently 200 chars
    - more overlap = less chance of splitting a value across chunks

## EMBEDDING
- converts chunks to vectors for similarity search
- embedding model: currently text-embedding-3-small @ 1536 dims (OpenAI)
    - could swap to local model (nomic-embed-text @ 768 dims) for cost savings
    - quality difference likely minimal for our use case (metadata does the heavy lifting)
- truncation: currently 2500 chars per chunk (to stay under 8192 token limit)

## VECTORDB
- chunk storage
- uses metadata (city, state, year) to automatically narrow search
- year search strategy: currently searches year AND year+1 (for biennial budgets)
- can we do anything else?

## QUERY PROMPT (retrieval)
- the text used to generate the embedding for vector search
- controls *which* chunks come back from the DB
- e.g. "General Fund expenditure" vs "total police budget"
- prompt caching: for efficiency and speed, not really for results
    - although if we are prompt caching, we should make sure we are updating the cache whenever we change the prompt!

## RETRIEVAL
- getting the candidate chunks from the vectorDB
- n-chunks: how many chunks do we return?
    - biggest lever here
    - too few chunks = might not get the right one
    - too many chunks = more noise for the LLM to handle
    - metric: how many chunks does it take to reach a threshold of coverage?

## LLM EXTRACTION PROMPT
- the system + user message sent to the LLM
- controls *how the model interprets* the chunks
- separate from the query prompt above
- system prompt: role, rules (expenditures only, adopted not proposed, etc.)
- user prompt: city/state/year/expense + chunk text

## LLM
- finds target value in the candidate chunks
- base model choice: currently Mistral 7B (could try Llama 3.1 8B, etc.)
- inference params:
    - max_tokens: currently 50 (caps output length)
    - temperature: currently 0 (deterministic)
- training data: lots of levers
    - size: currently at 517
    - parser balance: pymupdf v. aryn
        - most importantly, aryn tabular chunks
            - does aryn return any other types of novel chunks?
        - do pymupdf text chunks differ from aryn text chunks?
    - negative controls: teach the model what to NOT select
        - currently using structure caching to increase similarity to correct chunk
            - is this happening at the chunk or page level?
            - is this making the model worse (by teaching it to ignore "similar" chunks)?
            - positive to negative balance?
- training params (in finetune.py):
    - LoRA rank: currently r=16
    - LoRA target modules: all attention + MLP layers
    - learning rate: currently 2e-4
    - batch size: currently 2, gradient accumulation 4
    - epochs: currently 3
    - max_seq_length: currently 8192
    - package: unsloth + trl (SFTTrainer)
    - metrics: should be tracking loss per epoch

## EVALUATION
- test set composition: currently 108 records (6 cities x 2 expenses, expandable)
- scoring: exact integer match (extract_numbers)
- comparison fairness: must control n-chunks, max_tokens, etc. across runs
