"""RAG (Retrieval-Augmented Generation) tool for searching documentation.

This tool loads Markdown documents from the _docs directory, creates embeddings,
and provides a search interface for retrieving relevant documentation.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

from agent.tools.base import Tool, ToolContext, ToolResult


class RAGTool(Tool):
    """RAG tool for searching documentation using vector similarity search.
    
    This tool loads Markdown documents from the _docs directory, creates
    embeddings using HuggingFace models, stores them in Chroma, and provides
    a search operation to retrieve relevant documentation chunks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize RAG tool.
        
        Args:
            config: Tool configuration containing:
                - docs_path: Path to documentation directory (default: _docs)
                - embedding_model: HuggingFace model name (default: sentence-transformers/all-MiniLM-L6-v2)
                - persist_directory: Directory to persist Chroma DB (default: .chroma_db)
                - chunk_size: Text chunk size for splitting (default: 1000)
                - chunk_overlap: Overlap between chunks (default: 200)
                - k: Number of documents to retrieve (default: 4)
        """
        super().__init__(config)
        self.docs_path = Path(config.get("docs_path", "_docs"))
        self.embedding_model = config.get(
            "embedding_model", 
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.persist_directory = config.get("persist_directory", ".chroma_db")
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.k = config.get("k", 4)
        
        # Initialize components lazily
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vectorstore: Optional[Chroma] = None
        self._initialized = False
    
    @property
    def name(self) -> str:
        """Return tool name."""
        return "rag_documentation_search"
    
    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Search and retrieve relevant documentation from the codebase documentation. "
            "Use this tool when you need to find information about LangChain, tools, "
            "agents, or other topics covered in the documentation. "
            "Input should be a search query describing what information you're looking for."
        )
    
    def _ensure_initialized(self) -> None:
        """Initialize embeddings and vector store if not already done."""
        if self._initialized:
            return
        
        # Initialize embeddings
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        
        # Load or create vector store
        persist_path = Path(self.persist_directory)
        persist_path.mkdir(exist_ok=True)
        
        if persist_path.exists() and any(persist_path.iterdir()):
            # Load existing vector store
            try:
                self._vectorstore = Chroma(
                    persist_directory=str(persist_path),
                    embedding_function=self._embeddings
                )
                # Check if it has documents
                if self._vectorstore._collection.count() > 0:
                    self._initialized = True
                    return
            except Exception as e:
                print(f"Warning: Failed to load existing vector store: {e}")
                # Will create new one below
        
        # Create new vector store from documents
        self._load_documents()
        self._initialized = True
    
    def _load_documents(self) -> None:
        """Load Markdown documents from _docs directory and create vector store."""
        # Resolve path relative to project root if relative
        if not self.docs_path.is_absolute():
            # Try to find project root (where this file is located)
            project_root = Path(__file__).parent.parent.parent
            self.docs_path = (project_root / self.docs_path).resolve()
        
        if not self.docs_path.exists():
            raise ValueError(f"Documentation path does not exist: {self.docs_path}")
        
        # Find all Markdown files
        md_files = list(self.docs_path.glob("*.md"))
        if not md_files:
            raise ValueError(f"No Markdown files found in {self.docs_path}")
        
        # Load documents
        documents: List[Document] = []
        for md_file in md_files:
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": str(md_file),
                                "filename": md_file.name,
                                "title": md_file.stem
                            }
                        )
                    )
            except Exception as e:
                print(f"Warning: Failed to load {md_file}: {e}")
                continue
        
        if not documents:
            raise ValueError("No documents could be loaded")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        persist_path = Path(self.persist_directory)
        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            persist_directory=str(persist_path)
        )
        
        print(f"Loaded {len(documents)} documents ({len(chunks)} chunks) into vector store")
    
    async def execute(
        self,
        operation: str,
        arguments: Dict[str, Any],
        ctx: Optional[ToolContext] = None
    ) -> ToolResult:
        """Execute RAG search operation.
        
        Args:
            operation: Operation name (should be "search")
            arguments: Dictionary containing:
                - query: Search query string
                - k: Optional number of results (overrides default)
            ctx: Optional tool context
        
        Returns:
            ToolResult with search results
        """
        if not self.enabled:
            return ToolResult(
                success=False,
                output=None,
                error="RAG tool is disabled"
            )
        
        if operation != "search":
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown operation: {operation}. Supported: 'search'"
            )
        
        query = arguments.get("query")
        if not query or not isinstance(query, str):
            return ToolResult(
                success=False,
                output=None,
                error="Missing or invalid 'query' argument"
            )
        
        try:
            # Initialize if needed
            self._ensure_initialized()
            
            # Get number of results
            k = arguments.get("k", self.k)
            if not isinstance(k, int) or k <= 0:
                k = self.k
            
            # Perform similarity search
            results = self._vectorstore.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            
            return ToolResult(
                success=True,
                output={
                    "query": query,
                    "results": formatted_results,
                    "count": len(formatted_results)
                },
                metadata={
                    "k": k,
                    "model": self.embedding_model
                }
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"RAG search failed: {str(e)}"
            )


def create_rag_langchain_tool(rag_tool: RAGTool) -> Any:
    """Create a LangChain tool from RAGTool using @tool decorator.
    
    Args:
        rag_tool: RAGTool instance
    
    Returns:
        LangChain tool created with @tool decorator
    """
    @tool
    def rag_documentation_search(query: str, k: int = 4) -> str:
        """Search and retrieve relevant documentation from the codebase.
        
        Use this tool when you need to find information about LangChain, tools,
        agents, or other topics covered in the documentation.
        
        Args:
            query: Search query describing what information you're looking for
            k: Number of document chunks to retrieve (default: 4)
        
        Returns:
            JSON string with search results containing relevant documentation chunks
        """
        import asyncio
        import json
        
        # Execute the tool
        result = asyncio.run(
            rag_tool.execute(
                operation="search",
                arguments={"query": query, "k": k}
            )
        )
        
        if not result.success:
            return json.dumps({"error": result.error})
        
        # Format output for readability
        output = result.output
        formatted = f"Found {output['count']} relevant document(s) for query: '{output['query']}'\n\n"
        
        for i, res in enumerate(output['results'], 1):
            formatted += f"--- Result {i} (score: {res['score']:.4f}) ---\n"
            formatted += f"Source: {res['metadata'].get('filename', 'unknown')}\n"
            formatted += f"Content:\n{res['content']}\n\n"
        
        return formatted
    
    return rag_documentation_search
