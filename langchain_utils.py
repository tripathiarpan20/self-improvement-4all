from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.schema import Document
from typing import Any, Dict, List, Optional, Tuple
import faiss
import datetime
import os
import _pickle as cPickle
import bz2

#model_name = "intfloat/e5-base-v2"
model_name = "models/e5-base-v2"
model_kwargs = {'device': 'cpu'}
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs)
# Initialize the vectorstore as empty
embedding_size = 768

def _get_hours_passed(time: datetime, ref_time: datetime) -> float:
    """Get the hours passed between two datetime objects."""
    return (time - ref_time).total_seconds() / 3600

class TimeWeightedVectorStoreRetriever_custom(TimeWeightedVectorStoreRetriever):
    def _get_combined_score(
            self,
            document: Document,
            vector_relevance: Optional[float],
            current_time: datetime,
        ) -> float:
        """Return the combined score for a document."""
        hours_passed = _get_hours_passed(
            current_time,
            document.metadata["last_accessed_at"],
        )
        #Note: We can change the above 'last_accessed_at' above to 'created_at' to rank memory based on when it was created (rather than when it was last accessed in the Langchain default implementation)
        #https://github.com/hwchase17/langchain/blob/85dae78548ed0c11db06e9154c7eb4236a1ee246/langchain/retrievers/time_weighted_retriever.py#L119

        score = (1.0 - self.decay_rate) ** hours_passed
        # print(f'score contributed by time: {score}')
        for key in self.other_score_keys:
            if key in document.metadata:
                if key != 'importance':
                  score += document.metadata[key]
                else:
                  score += int(document.metadata[key])/10.
                #   print(f'score contributed by importance: {int(document.metadata[key])/10.}')

        if vector_relevance is not None:
            score += vector_relevance
            # print(f'score contributed by vector relevance: {vector_relevance}')

        # print(f'total score: {score}')
        # print('------------')
        return score

def prepare_memory_object(timestamp, lastAccess, vector, importance):
  memoryObject = {
        "timestamp": timestamp,
        "lastAccess": lastAccess,
        "vector": vector,
        "importance": importance
    }
  return memoryObject

def initialize_vector_memory():
  index = faiss.IndexFlatL2(embedding_size)
  vectorstore = FAISS(EMBEDDING_MODEL.embed_query, index, InMemoryDocstore({}), {})
  retriever = TimeWeightedVectorStoreRetriever_custom(vectorstore=vectorstore, other_score_keys = ['importance'] , decay_rate=.01, k=1)
  return retriever

def load_vector_memory(root = '/content/vector_mems', mem_name = 'dialogue'):
  path = os.path.join(root, mem_name) + '.pbz2'
  assert os.path.exists(path)
  data = bz2.BZ2File(path, "rb")
  retriever = cPickle.load(data)
  return retriever

def save_vector_memory(retriever, root = '/content/vector_mems', mem_name = 'dialogue'):
  path = os.path.join(root, mem_name) + '.pbz2'
  with bz2.BZ2File(path, "w") as f:
    cPickle.dump(retriever, f)
  return

def retrieve(query, retriever, top_k = 3):
  retriever.k = top_k
  retrieved_docs = retriever.get_relevant_documents(query)
  retrieved_docs_str = [('- ' + doc.page_content) for doc in retrieved_docs]
  return '\n'.join(retrieved_docs_str)