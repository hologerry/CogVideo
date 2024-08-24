# Video Caption

Typically, most video data does not come with corresponding descriptive text, so it is necessary to convert the video
data into textual descriptions to provide the essential training data for text-to-video models.

## Video Caption via CogVLM2-Video

🤗 [Hugging Face](https://huggingface.co/THUDM/cogvlm2-video-llama3-chat) | 🤖 [ModelScope](https://modelscope.cn/models/ZhipuAI/cogvlm2-video-llama3-chat) | 📑 [Blog](https://cogvlm2-video.github.io/) ｜ [💬 Online Demo](http://cogvlm2-online.cogviewai.cn:7868/)

CogVLM2-Video is a versatile video understanding model equipped with timestamp-based question answering capabilities.
Users can input prompts such as `Please describe this video in detail.` to the model to obtain a detailed video caption:

![CogVLM2-Video](./assests/cogvlm2-video-example.png)

Users can use the provided [code](https://github.com/THUDM/CogVLM2/tree/main/video_demo) to load the model or configure a RESTful API to generate video captions.
