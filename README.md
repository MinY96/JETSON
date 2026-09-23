(.venv) D:\개발\1_ImageProcessingTool\ImageProcessingAPI\backend>python scripts/prepare_openvino_sd15_inpaint.py --device CPU --compile-test
Multiple distributions found for package optimum. Picked distribution: optimum
`Siglip2ImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `Siglip2ImageProcessor` instead.
Source model : stable-diffusion-v1-5/stable-diffusion-inpainting
Output       : .image_processing_data\models\diffusion\sd15_openvino_inpaint\exported\stable-diffusion-v1-5__stable-diffusion-inpainting_512x512
Input size   : 512x512
D:\개발\1_ImageProcessingTool\ImageProcessingAPI\backend\.venv\Lib\site-packages\huggingface_hub\utils\_validators.py:205: UserWarning: The `local_dir_use_symlinks` argument is deprecated and ignored in `hf_hub_download`. Downloading to a local directory does not use symlinks anymore.
  warnings.warn(
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
model_index.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 548/548 [00:00<00:00, 552kB/s]
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 748/748 [00:00<00:00, 752kB/s]
Fetching 16 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [08:56<00:00, 33.53s/it]
Download complete: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 5.48G/5.48G [08:56<00:00, 11.4MB/s]Keyword arguments {'subfolder': ''} are not expected by StableDiffusionInpaintPipeline and will be ignored.
                                                                                                                                                                An error occurred while trying to fetch C:\Users\10007166\.cache\huggingface\hub\models--stable-diffusion-v1-5--stable-diffusion-inpainting\snapshots\8a4288a76071f7280aedbdb3253bdb9e9d5d84bb\vae: Error no file named diffusion_pytorch_model.safetensors found in directory C:\Users\10007166\.cache\huggingface\hub\models--stable-diffusion-v1-5--stable-diffusion-inpainting\snapshots\8a4288a76071f7280aedbdb3253bdb9e9d5d84bb\vae.
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
                                                                                                                                                                An error occurred while trying to fetch C:\Users\10007166\.cache\huggingface\hub\models--stable-diffusion-v1-5--stable-diffusion-inpainting\snapshots\8a4288a76071f7280aedbdb3253bdb9e9d5d84bb\unet: Error no file named diffusion_pytorch_model.safetensors found in directory C:\Users\10007166\.cache\huggingface\hub\models--stable-diffusion-v1-5--stable-diffusion-inpainting\snapshots\8a4288a76071f7280aedbdb3253bdb9e9d5d84bb\unet.
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 196/196 [00:00<00:00, 65343.26it/s]
CLIPTextModel LOAD REPORT from: C:\Users\10007166\.cache\huggingface\hub\models--stable-diffusion-v1-5--stable-diffusion-inpainting\snapshots\8a4288a76071f7280aedbdb3253bdb9e9d5d84bb\text_encoder                                                                                                      | 0/196 [00:00<?, ?it/s]
Key                                | Status     |  | 
-----------------------------------+------------+--+-
text_model.embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 396/396 [00:00<00:00, 26733.80it/s]
StableDiffusionSafetyChecker LOAD REPORT from: C:\Users\10007166\.cache\huggingface\hub\models--stable-diffusion-v1-5--stable-diffusion-inpainting\snapshots\8a4288a76071f7280aedbdb3253bdb9e9d5d84bb\safety_checker                                                                                     | 0/396 [00:00<?, ?it/s]
Key                                               | Status     |  | 
--------------------------------------------------+------------+--+-
vision_model.vision_model.embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Loading pipeline components...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:01<00:00,  4.67it/s]
Download complete: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 5.48G/5.48G [09:09<00:00, 9.98MB/s]
Writing model shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.25s/it]
`loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.
D:\개발\1_ImageProcessingTool\ImageProcessingAPI\backend\.venv\Lib\site-packages\transformers\models\clip\modeling_clip.py:243: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if seq_length > max_position_embedding:
`cache_position` is deprecated as an arg, and will be removed in Transformers v5.6. Please use `q_length` and `q_offset` instead, similarly to `kv_length` and `kv_offset`
D:\개발\1_ImageProcessingTool\ImageProcessingAPI\backend\.venv\Lib\site-packages\transformers\integrations\sdpa_attention.py:77: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  is_causal = query.shape[2] > 1 and attention_mask is None and is_causal
D:\개발\1_ImageProcessingTool\ImageProcessingAPI\backend\.venv\Lib\site-packages\diffusers\models\unets\unet_2d_condition.py:1050: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if dim % default_overall_up_factor != 0:
D:\개발\1_ImageProcessingTool\ImageProcessingAPI\backend\.venv\Lib\site-packages\diffusers\models\downsampling.py:134: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert hidden_states.shape[1] == self.channels
D:\개발\1_ImageProcessingTool\ImageProcessingAPI\backend\.venv\Lib\site-packages\diffusers\models\downsampling.py:143: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert hidden_states.shape[1] == self.channels
D:\개발\1_ImageProcessingTool\ImageProcessingAPI\backend\.venv\Lib\site-packages\diffusers\models\upsampling.py:145: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert hidden_states.shape[1] == self.channels
D:\개발\1_ImageProcessingTool\ImageProcessingAPI\backend\.venv\Lib\site-packages\diffusers\models\upsampling.py:160: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if hidden_states.shape[0] >= 64:
D:\개발\1_ImageProcessingTool\ImageProcessingAPI\backend\.venv\Lib\site-packages\diffusers\models\upsampling.py:169: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  2 if output_size is None else max([f / s for f, s in zip(output_size, hidden_states.shape[-2:])])
D:\개발\1_ImageProcessingTool\ImageProcessingAPI\backend\.venv\Lib\site-packages\diffusers\models\upsampling.py:171: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if hidden_states.numel() * scale_factor > pow(2, 31):
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 396/396 [00:00<00:00, 5653.68it/s]
