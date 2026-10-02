import asyncio
import httpx
from app.config import FAL_KEY
from app.providers import FAL_AUTO_URL


async def main():
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            FAL_AUTO_URL,
            headers={"Authorization": f"Key {FAL_KEY}"},
            json={"image_url": "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/notebooks/images/truck.jpg", "output_format": "png"},
        )
        print(response.status_code)
        print(response.text[:1200])


asyncio.run(main())
