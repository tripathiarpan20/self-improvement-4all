# Use the specified base image
FROM thebloke/cuda11.8.0-ubuntu22.04-oneclick:latest

# Set the working directory
WORKDIR /content

# Install required packages
RUN apt-get update && \
    apt-get -y install -qq aria2 && \
    apt-get -y install -qq git && \
    apt-get -y install -qq python3-pip && \
    apt-get -y install -qq python3

# Install setuptools
RUN python3 -m pip install --no-cache-dir --upgrade setuptools pip

# Clone the repository
RUN git clone https://github.com/oobabooga/text-generation-webui

# Set the working directory inside the cloned repository
WORKDIR /content/text-generation-webui

# Download the instruction templates
RUN wget -P instruction-templates/ https://raw.githubusercontent.com/tripathiarpan20/self-improvement-4all/main/prompt%20templates/Template_recalled_dialogue_2.yaml && \
    wget -P instruction-templates/ https://raw.githubusercontent.com/tripathiarpan20/self-improvement-4all/main/prompt%20templates/Template2.yaml

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir -U gradio==3.28.3 && \
    pip3 install --no-cache-dir sentence_transformers langchain==0.0.253 faiss-cpu==1.7.4 streamlit

# bitsandbytes debugging
RUN git clone https://github.com/TimDettmers/bitsandbytes.git 
WORKDIR /content/text-generation-webui/bitsandbytes
RUN CUDA_VERSION=120 make cuda11x
RUN python3 setup.py install
WORKDIR /content/text-generation-webui

# Download the model
RUN python3 download-model.py TheBloke/WizardLM-13B-V1.1-GPTQ

# Expose the required port (adjust if needed)
EXPOSE 7860
EXPOSE 5005
