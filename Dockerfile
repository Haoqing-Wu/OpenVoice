# Use the base image of Ubuntu
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Update the system and install necessary dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    sudo \
    python3.10 \
    python3-distutils \
    python3-pip \
    ffmpeg \
    git 

RUN apt install -y libcublas11

# Create a symlink for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Update pip
RUN pip3 install --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Install openai-whisper
RUN git clone https://github.com/myshell-ai/OpenVoice openvoice
COPY main.py /app/openvoice/openvoice/main.py

RUN pip3 install gradio==3.50.2 langid faster-whisper whisper-timestamped unidecode eng-to-ipa pypinyin cn2an

# Set the working directory in the container
WORKDIR /app/openvoice

RUN pip3 install -e .
RUN pip3 install soundfile librosa inflect jieba silero

RUN apt -y install -qq aria2 unzip

# Install Checkpoints for V1
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OpenVoice/resolve/main/checkpoints_1226.zip -d /app/openvoice -o checkpoints_1226.zip
RUN unzip /app/openvoice/checkpoints_1226.zip
RUN rm -f /app/openvoice/checkpoints_1226.zip
RUN mv /app/openvoice/checkpoints /app/openvoice/openvoice/checkpoints

# Install Checkpoints for V2
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip -d /app/openvoice -o checkpoints_v2_0417.zip
RUN unzip /app/openvoice/checkpoints_v2_0417.zip
RUN rm -f /app/openvoice/checkpoints_v2_0417.zip
RUN mv /app/openvoice/checkpoints_v2 /app/openvoice/openvoice/checkpoints 

# Move remaining files in place
RUN mv /app/openvoice/resources /app/openvoice/openvoice/resources 

# V2 specific installations
RUN pip3 install git+https://github.com/myshell-ai/MeloTTS.git
RUN python -m unidic download

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib
#EXPOSE 7860

#RUN sed -i "s/demo.launch(debug=True, show_api=True, share=args.share)/demo.launch(server_name='0.0.0.0', debug=True, show_api=True, share=args.share)/" /app/openvoice/openvoice/openvoice_app.py

WORKDIR /app/openvoice/openvoice

# Default command when the container is started
#CMD ["python3", "-m", "openvoice_app" ,"--share"]
