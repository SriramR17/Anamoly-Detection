# Anomaly Detection Dashboard - Integrated Frontend

This is the integrated frontend for the Anomaly Detection system, combining advanced UI components with real-time data integration from the backend API.

## Features

### 🎨 Modern UI Components
- **Interactive Dashboard**: Real-time metrics with animated visualizations
- **Advanced Charts**: Recharts-powered charts with animations and interactivity
- **Modern Design**: Tailwind CSS with dark theme and gradient effects
- **Responsive Layout**: Mobile-first design that works on all screen sizes

### 📊 Real Data Integration
- **API Integration**: Axios-based service for backend communication
- **Live Data**: Real-time fetching of dashboard metrics, predictions, and statistics
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Loading States**: Smooth loading animations and states

### 🔍 Key Sections

#### Dashboard
- Real-time statistics cards
- Interactive evaluation metrics with drill-down details
- Algorithm performance comparison charts
- Anomaly vs normal distribution visualization
- Peak hours analysis with gauge charts
- Network monitoring with severity indicators

#### Predictions Table
- Filterable and searchable prediction results
- Pagination for large datasets
- Detailed view modals with raw data
- Color-coded anomaly probability indicators
- Status badges for easy identification

#### Statistics
- Comprehensive statistical analysis
- Interactive pie and bar charts
- Probability statistics breakdown
- Key insights and recommendations
- Data quality indicators

### 🚀 Advanced Features
- **Run Detection**: Trigger anomaly detection directly from the UI
- **Auto-refresh**: Automatic data updates
- **Navigation**: Smooth tab-based navigation
- **Interactive Elements**: Hover effects, animations, and click interactions
- **TypeScript Support**: Full type safety and IntelliSense

## Technology Stack

- **React 18** - Modern React with hooks
- **TypeScript** - Type safety and better developer experience
- **Vite** - Fast build tool and development server
- **Tailwind CSS** - Utility-first CSS framework
- **Recharts** - Powerful charting library
- **Lucide React** - Beautiful icon library
- **Axios** - HTTP client for API requests
- **React Gauge Chart** - Gauge visualizations

## Getting Started

### Prerequisites
- Node.js 18+ 
- npm or yarn
- Backend API server running on port 8000

### Installation

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start the development server**:
   ```bash
   npm run dev
   ```

3. **Build for production**:
   ```bash
   npm run build
   ```

4. **Preview production build**:
   ```bash
   npm run preview
   ```

The application will be available at `http://localhost:5173`

### API Integration

The frontend expects the following API endpoints:

- `GET /api/dashboard` - Dashboard overview data
- `GET /api/predictions` - Prediction results
- `GET /api/stats` - Statistical analysis
- `POST /api/run-detection` - Trigger anomaly detection

The API proxy is configured to forward requests to `http://localhost:8000`.

## Project Structure

```
src/
├── components/           # React components
│   ├── PredictionsTable.tsx
│   └── Statistics.tsx
├── api.ts               # API service layer
├── types.d.ts           # Type definitions
├── App.tsx              # Main application component
├── main.tsx             # Application entry point
└── index.css            # Global styles
```

## API Response Formats

### Dashboard Data
```typescript
interface DashboardData {
  total_samples: number;
  total_anomalies: number;
  anomaly_rate: number;
  last_updated: string;
  recent_anomalies?: any[];
}
```

### Predictions
```typescript
interface Prediction {
  Predicted_Anomaly: number;
  Anomaly_Probability?: number;
  CellName?: string;
  Time?: string;
  [key: string]: any;
}
```

### Statistics
```typescript
interface Statistics {
  total_records: number;
  anomalies: number;
  normal: number;
  avg_anomaly_prob?: number;
  max_anomaly_prob?: number;
  min_anomaly_prob?: number;
}
```

## Customization

### Theme Colors
The dashboard uses a dark theme with accent colors defined in Tailwind CSS. You can customize colors by modifying the color classes in the components.

### Chart Configurations
Charts can be customized by modifying the Recharts configuration in each component. Colors, animations, and data formatting can be adjusted as needed.

### API Endpoints
Update the `API_BASE_URL` constant in `src/api.ts` to point to your backend server.

## Development Notes

- The application gracefully handles API failures by showing default/mock data
- All components are fully typed with TypeScript
- Responsive design breakpoints are optimized for mobile, tablet, and desktop
- Loading states and error handling are implemented throughout
- The build process generates optimized chunks for better performance

## Performance Optimizations

- Code splitting with dynamic imports
- Optimized bundle size with Vite's tree shaking
- Responsive image loading
- Debounced search inputs
- Virtualized large data lists (pagination)

This integrated dashboard provides a comprehensive view of your anomaly detection system with a modern, interactive interface that scales with your data.
