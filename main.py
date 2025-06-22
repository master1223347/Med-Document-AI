import cv2
import pytesseract
import openai

openai.api_key = "YOUR_OPENAI_API_KEY"

def prep(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def ocr(path):
    processed_img = prep(path)
    return pytesseract.image_to_string(processed_img)

def medsum(text):
    prompt = ("You are a helpful medical assistant. Given the following medical document text, "
              "explain in simple, easy-to-understand language what the doctor said about the "
              "patient’s symptoms, diagnosis, and any recommendations. Avoid medical jargon.\n\n"
              + text)
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()

path = " " #add file name here
text = ocr(path)
print(text)
print(medsum(text))
