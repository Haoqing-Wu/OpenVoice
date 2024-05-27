import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

import requests
import json
import subprocess

import time
import base64
from base64 import b64decode

from io import BytesIO
from pydantic import BaseModel, Field
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

def openvoice_v2(reference_speaker_path, text, language, speaker, speed):
    ckpt_converter = 'checkpoints/checkpoints_v2/converter'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = 'outputs_v2'

    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    os.makedirs(output_dir, exist_ok=True)

    reference_speaker = 'resources/example_reference.mp3' # This is the voice you want to clone
    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)




    src_path = f'{output_dir}/tmp.wav'
    
    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id
    speaker_key = speaker
    speaker_id = speaker_ids[speaker_key]
    speaker_key = speaker_key.lower().replace('_', '-')
    
    source_se = torch.load(f'checkpoints/checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
    model.tts_to_file(text, speaker_id, src_path, speed=speed)
    save_path = f'{output_dir}/output.wav'

    # Run the tone color converter
    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=save_path,
        message=encode_message)



app = FastAPI(title="openvoice_v2",version='0.0.1')


origins = ["*"]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"])  

class Parameters(BaseModel):
    debug: bool = Field(title="Debug", type="boolean", description="provide debugging output in logs", default=True)
    ref_audio: str = Field(title="Reference audio", type="string", description="Reference audio for voice cloning, in url format")
    text: str = Field(title="Text", type="string", description="Text to generate voice from reference")
    language: str = Field(title="Language", type="string", description="Language of the output (EN, ES, FR, ZH, JP, KR)", default="EN")
    speaker: str = Field(title="Speaker", type="string", description="Speaker to use for the output, choose from (EN-US, EN-BR, EN-INDIA, EN-AU, EN-Default) if language is EN, keep same wíth language if not EN", default="EN-Default")
    speed: float = Field(title="Speed", type="float", description="Speed of the output audio, default 1.0", default=1.0)
    

class InputData(BaseModel):
    input: Parameters = Field(title="Input")

def audio_to_data_uri(filename):
    ext = filename.split('.')[-1]
    prefix = f'data:audio/{ext};base64,'
    with open(filename, 'rb') as f:
        data = f.read()
    return prefix + base64.b64encode(data).decode('utf-8')

@app.get("/health-check")
def health_check():
    return {"status": "True"}

@app.post("/predictions")
def aigic(inputdata: InputData):
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    print('1. Input Data processing...')

    # Prepare the data 
    
    response = requests.get(inputdata.input.ref_audio)
    ext = inputdata.input.ref_audio.split('.')[-1]
    reference_speaker_path = 'resources/ref_input.' + ext
    with open(reference_speaker_path, 'wb') as f:
        f.write(response.content)
    

    print('2. Running the model...')
    # Run the model
    openvoice_v2(reference_speaker_path, inputdata.input.text, inputdata.input.language, inputdata.input.speaker, inputdata.input.speed)


    # Read the output
    output = audio_to_data_uri('outputs_v2/output.wav')


    print('3. Output')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')

    if len(output) == 0:
        return {
        #"code": 400,
        #"msg": "error",
        "output": "No output"
        }


    return {
    #"code": 200,
    #"msg": "success",
    "output": output 
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host="0.0.0.0", port=5001)