# Some models can return multimodal data as part of their response. 
# If invoked to do so, the resulting AIMessage will have content blocks with multimodal types.

response = model.invoke("Create a picture of a cat")
print(response.content_blocks)
# [
#     {"type": "text", "text": "Here's a picture of a cat"},
#     {"type": "image", "base64": "...", "mime_type": "image/jpeg"},
# ]