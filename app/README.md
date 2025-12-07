# YouTube Summarizer

**University of Michigan Capstone Project**  
**Team Candles**

## Overview

YouTube Summarizer is an AI-powered web application that generates intelligent summaries of YouTube videos. Simply paste a YouTube URL and receive a concise summary of the video's content in seconds.

## Features

- 🎥 **Instant Summaries**: Paste any YouTube URL to get an AI-generated summary
- 🎨 **University of Michigan Themed**: Custom design featuring Maize and Blue colors
- ⚡ **Real-time Processing**: Handles long-running API requests with appropriate feedback
- 📋 **Copy to Clipboard**: Easily copy summaries for later use
- 🌐 **Responsive Design**: Works seamlessly on desktop and mobile devices

## Technology Stack

- **Frontend**: React, TypeScript, Vite
- **UI Framework**: shadcn-ui components with Tailwind CSS
- **Backend**: Lovable Cloud (Supabase) with Edge Functions
- **AI Processing**: External summarization API

## Getting Started

### Prerequisites

- Node.js & npm ([install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating))

### Installation

1. Clone the repository:
```sh
git clone <YOUR_GIT_URL>
cd <YOUR_PROJECT_NAME>
```

2. Install dependencies:
```sh
npm install
```

3. Start the development server:
```sh
npm run dev
```

## How It Works

1. User enters a YouTube URL in the input field
2. Frontend sends the URL to the backend edge function
3. Edge function calls the external summarization API with proper authentication
4. AI processes the video and generates a summary
5. Summary is displayed to the user with copy functionality

## API Integration

The application integrates with an external summarization API:

- **Endpoint**: `http://34.229.218.179:8000/summarize`
- **Method**: POST
- **Authentication**: API Key via header
- **Timeout**: 2 minutes (accounting for long processing times)

## Project Structure

```
├── src/
│   ├── components/        # Reusable UI components
│   ├── pages/            # Page components
│   ├── integrations/     # Supabase client configuration
│   └── index.css         # Global styles and design tokens
├── supabase/
│   └── functions/        # Edge functions for backend logic
└── public/               # Static assets
```

## Deployment

This project is deployed on Lovable. To publish updates:

1. Click the **Publish** button in the Lovable editor
2. Click **Update** to deploy frontend changes
3. Backend changes (edge functions) deploy automatically

## Team

**Team Candles** - University of Michigan Capstone Project

## License

Built with [Lovable](https://lovable.dev)
