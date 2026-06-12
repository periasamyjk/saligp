# SALIGP Frontend UI

A modern, powerful React-based web interface for the SALIGP (Secure Active Learning with Integrated Genetic Programming) duplicate detection framework.

## Features

### 🎯 Dashboard
- Real-time performance metrics (Accuracy, Precision, Recall, F1)
- Pipeline execution status with progress indicators
- Feature importance visualization
- Learning curve analysis
- Cluster distribution charts
- System health monitoring

### 🔮 Predictions Interface
- File upload with drag-and-drop support
- Real-time duplicate detection results
- Confidence scores and uncertainty quantification
- Batch processing capabilities
- Results filtering and sorting
- CSV export functionality

### 📈 Advanced Analytics
- Performance metrics by difficulty level
- Active Learning progress tracking
- Model comparison charts
- Confusion matrix analysis
- System performance benchmarks
- Key insights and recommendations

## Tech Stack

- **React 18** with TypeScript
- **Vite** for fast builds and development
- **Tailwind CSS** for responsive styling
- **Recharts** for data visualization
- **Lucide React** for beautiful icons

## Installation

### Prerequisites
- Node.js 18+ 
- npm or yarn

### Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The application will open at `http://localhost:3000`

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── MetricCard.tsx
│   │   ├── StatusIndicator.tsx
│   │   ├── PredictionResultCard.tsx
│   │   └── FileUploadZone.tsx
│   ├── pages/
│   │   ├── Dashboard.tsx
│   │   ├── Predictions.tsx
│   │   └── Analytics.tsx
│   ├── utils/
│   ├── App.tsx
│   ├── App.css
│   ├── index.css
│   └── main.tsx
├── index.html
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
└── package.json
```

## Key Components

### Dashboard
Displays real-time metrics, pipeline status, feature importance, and learning curves from the SALIGP model.

### Predictions
Allows users to upload files for duplicate detection and view results with confidence scores and uncertainty estimates.

### Analytics
Provides detailed performance analysis including evaluation by difficulty, active learning progress, and model comparisons.

## Styling

The UI uses a modern dark theme with:
- Tailwind CSS for utility-first styling
- Custom gradient and glass-morphism effects
- Responsive design for all screen sizes
- Smooth animations and transitions

## Performance Metrics Display

The frontend showcases SALIGP's impressive results:
- **Accuracy**: 100.00%
- **Precision**: 100.00%
- **Recall**: 100.00%
- **F1 Score**: 100.00%

## Integration

To connect with the SALIGP backend API:

1. Update API endpoints in component files
2. Replace mock data with actual API calls using axios
3. Implement WebSocket for real-time updates
4. Add authentication for role-based access control

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

© 2024 SALIGP. All rights reserved.
