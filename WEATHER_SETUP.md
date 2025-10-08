# Weather API Setup

## Overview
The app now supports live weather integration in the Build Outfit tab using the Visual Crossing Weather API.

## Setup Instructions

### 1. Get Your API Key
1. Go to https://www.visualcrossing.com/
2. Sign up for a free account
3. Copy your API key

### 2. Add to Environment Variables
Add the following line to your `.env` file:
```
WEATHER_API_KEY=your_api_key_here
```

### 3. How to Use

#### In the Build Outfit Tab:
1. Enter a city name in the "City" field (e.g., "New York", "London", "Tokyo")
2. Click the **🌡️ Get Weather** button
3. Current weather will be displayed below the city input
4. Click **Generate Outfit** to get outfit suggestions based on:
   - Your wardrobe items
   - Selected occasion
   - Season
   - **Live weather conditions** (temperature and conditions)

## Features

### Weather Display
When you fetch weather, you'll see:
- 🌡️ Current temperature (°C by default)
- Weather conditions (e.g., "Partially cloudy", "Clear", "Rain")

### Smart Outfit Generation
The LLM will consider:
- **Actual temperature** instead of just season
- **Weather conditions** (rain, snow, clouds, etc.)
- **Location-specific factors** (humidity, wind, etc.)

### Example Usage:
**Without Weather:**
- City: "Stockholm"
- Season: "Spring"
- Result: Generic spring outfit

**With Weather:**
- City: "Stockholm" → Click "Get Weather"
- Shows: "🌡️ 8°C, Light rain"
- Result: Spring outfit adapted for rain and cool weather (jacket, closed shoes, etc.)

## API Details

- **Provider:** Visual Crossing Weather API
- **Free Tier:** 1000 requests/day
- **Default Unit:** Metric (°C)
- **Timeout:** 10 seconds

## Troubleshooting

### "Weather data unavailable"
- Check your API key in `.env`
- Verify the city name is correct
- Check your internet connection
- Free tier may have rate limits

### Weather not updating outfit
- Make sure to fetch weather BEFORE clicking "Generate Outfit"
- The weather data is passed to the LLM for better recommendations

## Privacy & Security
- API key is stored in `.env` (not committed to git)
- Weather data is fetched on-demand (not stored)
- No personal data is sent to the weather API
