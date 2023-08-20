# Use the specified base image
FROM runpod/pytorch:3.10-1.13.1-116

# Set the working directory
WORKDIR /content

# Install required packages
RUN apt-get update && \
    apt-get -y install -qq aria2 && \
    apt-get -y install -qq git && \
    apt-get -y install -qq python3-pip

# Install setuptools
RUN python3 -m pip install --no-cache-dir --upgrade setuptools pip

# Clone the repository
RUN git clone https://github.com/oobabooga/text-generation-webui

# Set the working directory inside the cloned repository
WORKDIR /content/text-generation-webui

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir -U gradio==3.28.3 && \
    pip3 install --no-cache-dir sentence_transformers langchain==0.0.253 faiss-cpu==1.7.4 streamlit

# Download the model
RUN python3 download-model.py TheBloke/WizardLM-13B-V1.1-GPTQ

# Expose the required port (adjust if needed)
EXPOSE 7860

# Start the server
CMD ["python3", "server.py", "--api", "--wbits", "4", "--groupsize", "128", "--model", "TheBloke_WizardLM-13B-V1.1-GPTQ", "--model_type", "llama", "--loader", "exllama"]
