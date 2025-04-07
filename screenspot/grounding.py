import io
import json
import base64

PROMPT = """You are specialize in GUI grounding. This is an interface to a GUI and you are going to perform click action to {instruction}.
Output exactly one line containing a single JSON object in the following format:
{{"point_2d": [x, y], "label": "object name/description"}}"""

class Grounding:
    def __init__(self, client, model=None):
        self.client = client
        self.model = model if model else client.models.list().data[0].id
        print(f"Grounding model: {self.model}")

    def call(self, instruction, image_path):
        """
        call the model to locate the element,
        return x, y
        """
        base64_image = encode_image(image_path)
        messages = get_grounding_messages(instruction, base64_image)
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=512,
            temperature=0
        )
        response = completion.choices[0].message.content
        # print("Response: " + response)
        return self.parse_coordinates(response)
    
    def parse_coordinates(self, response):
        """
        Parse JSON format response to extract coordinates. Example:
        {"point_2d": [x, y], "label": "object name/description"}
        
        Returns:    
            tuple: (x, y) coordinates
        """
        # 去除可能的代码块标记
        if response.startswith("```json"):
            response = response[7:].strip()
        if response.endswith("```"):
            response = response[:-3].strip()
        
        try:
            data = json.loads(response)
            coords = data.get("point_2d")
            if isinstance(coords, list) and len(coords) == 2:
                return tuple(coords)
            else:
                print("JSON 中 'point_2d' 格式不符合预期, 错误格式输出: " + response)
                return None, None
        except json.JSONDecodeError as e:
            print("JSON 格式错误, 错误格式输出: " + response)
            return None, None


def get_grounding_messages(instruction, base64_image):
    instruction = PROMPT.format(instruction=instruction)
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    },
                },
                {
                    "type": "text",
                    "text": instruction
                },
            ],
        },
    ]
    return messages


def encode_image(image_path):
    # encode image to base64 string
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    encoded_image_text = encoded_image.decode("utf-8")
    return encoded_image_text
