# ONDC Vector Search Demo UI

A sophisticated React/Next.js interface for demonstrating AI-powered semantic search capabilities on ONDC catalogs.

## Features

🎨 **Enhanced Design**
- Compelling hero section with gradient backgrounds
- Sophisticated search interface with ONDC green branding
- Confidence indicators and match explanations
- Responsive design for all device sizes

🚀 **AI-Powered Search**
- Vector embeddings for semantic similarity
- Cross-encoder reranking for quality
- Intent parsing and constraint handling
- Real-time search with performance metrics

🎯 **User Experience**
- Interactive result cards with expandable score breakdowns
- Sample query suggestions
- Smooth animations and transitions
- Progressive disclosure of technical details

## Tech Stack

- **Framework**: Next.js 15 with TypeScript
- **Styling**: TailwindCSS with custom design tokens
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **API Integration**: FastAPI backend integration

## Getting Started

### Prerequisites

- Node.js 18+ installed
- FastAPI backend running on `http://localhost:8080`

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

### Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8080
```

## API Integration

The UI connects to your FastAPI backend with the following endpoints:

- `GET /search` - Main search functionality
- `GET /suggest` - Autocomplete suggestions
- `POST /ingest` - Data ingestion (admin)

## Deployment

### Vercel (Recommended)

1. Connect your GitHub repository to Vercel
2. Set the environment variable: `NEXT_PUBLIC_API_BASE_URL`
3. Deploy automatically on push

### Manual Build

```bash
npm run build
npm start
```

## Project Structure

```
src/
├── app/
│   ├── globals.css      # Global styles and design tokens
│   ├── layout.tsx       # Root layout
│   └── page.tsx         # Main page component
├── components/
│   ├── SearchInterface.tsx  # Main search interface
│   └── ResultCard.tsx       # Enhanced result cards
└── lib/
    └── api.ts           # API integration utilities
```

## Design System

### Colors
- **ONDC Green**: `#0A8F47` - Primary actions, high confidence
- **Success Green**: `#16A34A` - Vegetarian indicators, positive feedback  
- **Warning Orange**: `#EA580C` - Medium confidence, non-veg indicators
- **Neutral Gray**: `#6B7280` - Secondary text, borders

### Typography
- **Font**: Inter (Google Fonts)
- **Sizes**: 32px (hero) → 18px (subtitle) → 16px (cards) → 14px (metadata)

### Components
- **Hero Gradient**: `linear-gradient(135deg, #FFFFFF 0%, #F0FDF4 100%)`
- **Card Shadows**: Elevated with hover animations
- **Confidence Indicators**: Color-coded progress bars and badges

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the FastAPI backend
5. Submit a pull request

## License

This project is part of the ONDC ecosystem demonstration.
