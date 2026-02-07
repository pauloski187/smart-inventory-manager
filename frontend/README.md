# Smart Inventory Manager - Frontend

A modern, responsive inventory management dashboard built with **Next.js 14**, **TypeScript**, and **Tailwind CSS**. Connects to a live FastAPI backend on Hugging Face Spaces.

## 🚀 Live Demo

- **Frontend**: Deploy to Vercel (instructions below)
- **Backend API**: https://pauloski07-sales-inventory-forecasting.hf.space
- **API Docs**: https://pauloski07-sales-inventory-forecasting.hf.space/docs

## ✨ Features

- ✅ **Dashboard** - KPIs and metrics overview
- ✅ **Sales Trends** - Revenue and profit analysis with charts
- ✅ **Category Performance** - Revenue share and performance metrics
- ✅ **Product Performance** - Best and worst performing products
- ✅ **Inventory Alerts** - Low stock and dead stock monitoring
- ⏳ **Additional Pages** - Templates provided for expansion

## 🛠️ Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Icons**: Lucide React
- **Deployment**: Vercel

## 📦 Installation

### Prerequisites

- Node.js 18+
- npm or yarn

### Step 1: Install Dependencies

```bash
cd frontend
npm install
```

### Step 2: Environment Configuration

The API URL is already configured in `.env.local`:

```env
NEXT_PUBLIC_API_URL=https://pauloski07-sales-inventory-forecasting.hf.space
```

No changes needed!

### Step 3: Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 🚀 Deploy to Vercel (Production)

### Method 1: Vercel CLI (Recommended)

1. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy**:
   ```bash
   cd frontend
   vercel
   ```

4. **Follow the prompts**:
   - Set up and deploy? **Yes**
   - Which scope? Choose your account
   - Link to existing project? **No**
   - Project name? **smart-inventory-frontend** (or your choice)
   - Directory? **./** (current directory)
   - Override settings? **No**

5. **Production deployment**:
   ```bash
   vercel --prod
   ```

Your app will be live at: `https://your-project-name.vercel.app`

### Method 2: GitHub + Vercel (Continuous Deployment)

1. **Push to GitHub**:
   ```bash
   cd frontend
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. **Connect to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will auto-detect Next.js
   - Click "Deploy"

3. **Configure Environment Variables** (optional):
   - In Vercel dashboard → Settings → Environment Variables
   - Add: `NEXT_PUBLIC_API_URL` = `https://pauloski07-sales-inventory-forecasting.hf.space`

4. **Done!** Every push to `main` branch will auto-deploy.

### Method 3: Manual Vercel Dashboard

1. **Build locally**:
   ```bash
   npm run build
   ```

2. **Deploy**:
   - Go to [vercel.com/new](https://vercel.com/new)
   - Drag and drop the `frontend` folder
   - Click "Deploy"

## 🌐 Custom Domain (Optional)

### Add Your Own Domain

1. **In Vercel Dashboard**:
   - Go to your project
   - Settings → Domains
   - Add your domain (e.g., `inventory.yourdomain.com`)

2. **Update DNS**:
   - Add CNAME record pointing to Vercel
   - Vercel provides instructions

3. **SSL**: Automatic (provided by Vercel)

## 📁 Project Structure

```
frontend/
├── src/
│   ├── app/              # Next.js App Router pages
│   │   ├── dashboard/    # Dashboard page
│   │   ├── trends/       # Sales trends page
│   │   ├── categories/   # Category performance
│   │   ├── products/     # Product performance
│   │   ├── alerts/       # Inventory alerts
│   │   └── ...           # Other pages (templates)
│   ├── components/
│   │   ├── ui/           # Reusable UI components
│   │   ├── charts/       # Chart components
│   │   └── layout/       # Layout components (Sidebar)
│   ├── lib/
│   │   ├── api.ts        # API integration layer
│   │   └── utils.ts      # Utility functions
│   └── types/
│       └── api.ts        # TypeScript types
├── public/               # Static assets
├── .env.local           # Environment variables
├── next.config.js       # Next.js configuration
├── tailwind.config.ts   # Tailwind CSS config
├── tsconfig.json        # TypeScript configuration
└── package.json         # Dependencies
```

## 🎨 Customization

### Update API URL

Edit `.env.local`:
```env
NEXT_PUBLIC_API_URL=https://your-api-url.com
```

### Change Colors

Edit `tailwind.config.ts`:
```typescript
colors: {
  primary: { DEFAULT: "#3B82F6" },  // Change to your brand color
  success: { DEFAULT: "#10B981" },
  // ...
}
```

### Add New Pages

1. Create new file in `src/app/your-page/page.tsx`
2. Add route to Sidebar in `src/components/layout/Sidebar.tsx`
3. Use existing page templates as reference

## 🔧 Development

### Available Scripts

```bash
npm run dev      # Start development server (localhost:3000)
npm run build    # Build for production
npm run start    # Start production server
npm run lint     # Run ESLint
```

### Adding a New Chart

1. Create component in `src/components/charts/`
2. Use Recharts library
3. Import in your page

Example:
```tsx
import { LineChart } from "@/components/charts/LineChart";

<LineChart
  data={chartData}
  xKey="month"
  lines={[
    { key: "revenue", color: "#3B82F6", name: "Revenue" },
  ]}
/>
```

## 🐛 Troubleshooting

### Build Errors

**Problem**: `Module not found` errors

**Solution**:
```bash
rm -rf node_modules package-lock.json
npm install
```

### API Connection Issues

**Problem**: CORS errors or API not responding

**Solution**:
1. Check API is live: https://pauloski07-sales-inventory-forecasting.hf.space/health
2. Verify `.env.local` has correct URL
3. Restart dev server: `npm run dev`

### Deployment Fails on Vercel

**Problem**: Build fails

**Solution**:
1. Check build locally: `npm run build`
2. Fix any TypeScript errors
3. Ensure all dependencies are in `package.json`
4. Try: `npm install --legacy-peer-deps`

## 📊 Expanding the Application

### Add More Pages

Templates are provided for:
- ABC Analysis (`/abc-analysis`)
- Sales Patterns (`/patterns`)
- Monthly Reports (`/reports`)
- Products List (`/products-list`)
- Orders List (`/orders-list`)

To complete these pages:
1. Open the page file (e.g., `src/app/patterns/page.tsx`)
2. Use Dashboard or Trends page as a reference
3. Call the appropriate API endpoint
4. Display the data in tables/charts

### Example: Complete the Patterns Page

```tsx
// src/app/patterns/page.tsx
import { api } from "@/lib/api";
import { Card } from "@/components/ui/Card";

async function PatternsContent() {
  const data = await api.getSalesByDayOfWeek();

  return (
    <Card>
      {/* Display day of week analysis */}
    </Card>
  );
}

export default function PatternsPage() {
  return (
    <div>
      <h1>Sales Patterns</h1>
      <Suspense fallback={<Loading />}>
        <PatternsContent />
      </Suspense>
    </div>
  );
}
```

## 🚀 Performance Optimization

### Built-in Optimizations

- ✅ Server-side rendering (SSR)
- ✅ API response caching (60 seconds)
- ✅ Code splitting
- ✅ Image optimization
- ✅ Lazy loading

### Further Improvements

1. **Add React Query** for better data fetching:
   ```bash
   npm install @tanstack/react-query
   ```

2. **Enable Static Generation** for some pages
3. **Add loading skeletons** for better UX
4. **Implement pagination** for large data sets

## 📝 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API base URL | HF Spaces URL |

## 🔒 Security

- ✅ Environment variables for sensitive data
- ✅ CORS handled by backend
- ✅ No authentication required (demo mode)
- ⚠️ For production: Add authentication/authorization

## 📚 Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [Recharts](https://recharts.org)
- [Vercel Deployment](https://vercel.com/docs)

## 🤝 Support

Having issues?

1. Check [Troubleshooting](#-troubleshooting) section
2. Review [Backend API Docs](https://pauloski07-sales-inventory-forecasting.hf.space/docs)
3. Ensure backend is online: [Health Check](https://pauloski07-sales-inventory-forecasting.hf.space/health)

## 📄 License

MIT License - Free to use for personal and commercial projects

---

## ✅ Deployment Checklist

Before deploying to production:

- [ ] Install dependencies: `npm install`
- [ ] Test locally: `npm run dev`
- [ ] Build successfully: `npm run build`
- [ ] Check API connection works
- [ ] Create Vercel account
- [ ] Deploy using one of the methods above
- [ ] Test production deployment
- [ ] (Optional) Add custom domain
- [ ] Share your live URL!

---

**Your Setup:**

- Frontend: Deploy to Vercel ➡️ `your-app.vercel.app`
- Backend: Already live on HF Spaces ✅
- Result: Professional, production-ready inventory management system!

🎉 **Ready to deploy!**
