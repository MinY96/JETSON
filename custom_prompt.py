PROMPT = (
    "close-up of a flexible industrial hose near its connector, "
    "a small realistic tear in the hose surface, "
    "a strong pressurized stream of clear water leaking directly from the torn area, "
    "visible continuous water jet with small droplets and wet reflections around the damaged hose, "
    "realistic industrial CCTV image, "
    "same hose, same connector, same equipment, same lighting and camera perspective, "
    "localized damage and water leakage only, "
    "photorealistic"
)

PROMPT = (
    "close-up of a flexible industrial hose near its connector, "
    "a small realistic tear in the hose surface, "
    "a strong pressurized stream of clear water leaking directly from the torn area, "
    "a clearly visible transparent water jet with bright specular highlights, "
    "small airborne droplets and glossy wet reflections around the damaged hose, "
    "realistic industrial CCTV image, "
    "same hose, same connector, same equipment, "
    "same lighting and camera perspective, "
    "localized damage and water leakage only, "
    "photorealistic"
)

NEGATIVE_PROMPT = (
    "extra hoses, extra connectors, extra pipes, new objects, "
    "hands, people, tools, text, labels, "
    "different equipment, changed background, different camera angle, "
    "large broken hose, completely severed hose, exploded hose, "
    "flooding, huge splash, smoke, fire, "
    "deformed connector, distorted machinery, "
    "cartoon, CGI, illustration, blurry, low quality"
)

"parameters": {
    "model": "sd15_openvino_inpaint",

    "prompt": PROMPT,

    "negative_prompt": NEGATIVE_PROMPT,

    "device": DEVICE,

    "input_size": 512,

    "steps": STEPS,

    "guidance_scale": GUIDANCE_SCALE,

    "strength": STRENGTH,
},


STEPS = 20
GUIDANCE_SCALE = 7.5
STRENGTH = 0.85
SEED = 100
DEVICE = "GPU"
