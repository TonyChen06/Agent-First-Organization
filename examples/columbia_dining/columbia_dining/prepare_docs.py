"""Prepare local references for Arklex's FAISS retriever."""

import json
import pickle
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def prepare_documents(config: dict, output_dir: Path) -> list[Document]:
    # Match Arklex 0.1.21's Loader.chunk settings without its OCR dependencies.
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=200, chunk_overlap=40
    )
    chunks = []
    for reference in config["rag_docs"]:
        path = Path(reference["source"])
        title, source_line, body = path.read_text(encoding="utf-8").split("\n", 2)
        if not source_line.startswith("Source: https://") or not body.strip():
            raise ValueError(f"Expected a source URL and reference text in {path}")
        source = source_line.removeprefix("Source: ").strip()
        for passage in splitter.split_text(body.strip()):
            chunks.append(
                Document(
                    page_content=f"{title}\nSource: {source}\n\n{passage}",
                    metadata={"title": title, "source": source},
                )
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "chunked_documents.pkl").open("wb") as output:
        pickle.dump(chunks, output)
    return chunks


def main() -> None:
    config = json.loads(Path("columbia_dining/config.json").read_text())
    output_dir = Path("columbia_dining/data")
    chunks = prepare_documents(config, output_dir)
    print(f"Saved {len(chunks)} passages to {output_dir / 'chunked_documents.pkl'}")


if __name__ == "__main__":
    main()
