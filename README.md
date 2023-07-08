# self-improv-4-all

This project is done under the partial fulfilment of MSc AI & ML course at The University of Birmingham.

#Background

In the contemporary world with an uncertain future and the widespread sense of self-alienation
among modern people, there is a necessity of universal support for self-improvement. This
project aims to make such support more accessible by using the intrinsic knowledge and
empathy towards users present in open-source chat-based large language models, with a focus
on user-data privacy.

Under the project, the objectives/implemented features are as follows:
- [ ] Implementation of Plan-Act-Reflect pardigm from [Generative Agents paper](https://arxiv.org/abs/2304.03442) by Stanford to simulate a proxy of a self-improvement coach, by utilising a backend LLM.
- [ ] Compatibility with future open-source/LLaMa-based models via integration with Huggingface, assuming sufficient alignment, instruction-following capabilities and knowledge of psychology in the LLM.
- [ ] Support for deploying backend LLM locally with GPU acceleration using [GPTQ](https://github.com/IST-DASLab/gptq)/[GGML](http://ggml.ai/) based models with Huggingface, accessible via API, using [text-generation-inference](https://github.com/huggingface/text-generation-inference) by Huggingface to boost performance.
- [ ] Support for deploying backend LLM on cloud services, specifically, using [inference-endpoints](https://huggingface.co/inference-endpoints).
- [ ] Support for a simplistic chat UI based on Streamlit.



#References:
- [Langchain](https://github.com/hwchase17/langchain)
- [Generative Agents](https://arxiv.org/abs/2304.03442)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [FAISS](https://github.com/facebookresearch/faiss)
- [GPTQ](https://github.com/IST-DASLab/gptq)
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [text-generation-inference](https://github.com/huggingface/text-generation-inference)

#Scope for Future work:
- [ ] Enhancing privacy protection by deploying text tokenizer on the frontend and transmit token indices' sequence in API calls, assuming token indices to be unique to the model.
- [ ] Enhancing defence against prompt injection attacks for added safety.
