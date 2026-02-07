# 🚀 Deployment Guide - Smart Inventory Frontend

## Quick Start (5 Minutes)

### 1. Test Locally First

```bash
cd /Users/paulolusola/PyCharmMiscProject/smart-inventory-manager/frontend
npm install
npm run dev
```

Visit: http://localhost:3000

**Expected**: You should see the dashboard with data from the live API!

### 2. Deploy to Vercel (Fastest Method)

```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy
vercel

# Production deployment
vercel --prod
```

**Done!** Your app is live in ~2 minutes!

---

## Detailed Deployment Options

### Option A: Vercel CLI (Recommended) ⭐

**Why**: Fastest, easiest, most reliable

**Steps**:

1. Install Vercel CLI:
   ```bash
   npm install -g vercel
   ```

2. Navigate to project:
   ```bash
   cd /Users/paulolusola/PyCharmMiscProject/smart-inventory-manager/frontend
   ```

3. Login to Vercel:
   ```bash
   vercel login
   ```

   Choose login method (GitHub, email, etc.)

4. Deploy:
   ```bash
   vercel
   ```

5. Answer prompts:
   ```
   ? Set up and deploy? Yes
   ? Which scope? [Your account]
   ? Link to existing project? No
   ? What's your project's name? smart-inventory-frontend
   ? In which directory is your code located? ./
   ? Want to override the settings? No
   ```

6. Preview deployment:
   - Vercel provides a preview URL
   - Test it: `https://your-project-name-xxx.vercel.app`

7. Deploy to production:
   ```bash
   vercel --prod
   ```

8. **Done!** Production URL: `https://your-project-name.vercel.app`

---

### Option B: GitHub + Vercel (Continuous Deployment)

**Why**: Auto-deploys on every git push

**Steps**:

1. **Create GitHub Repo**:
   - Go to github.com/new
   - Name: `smart-inventory-frontend`
   - Public or Private
   - Create repository

2. **Push Code**:
   ```bash
   cd /Users/paulolusola/PyCharmMiscProject/smart-inventory-manager/frontend

   git init
   git add .
   git commit -m "Initial commit - Smart Inventory Frontend"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/smart-inventory-frontend.git
   git push -u origin main
   ```

3. **Connect to Vercel**:
   - Go to [vercel.com/new](https://vercel.com/new)
   - Click "Import Project"
   - Select your GitHub repository
   - Vercel auto-detects Next.js config
   - Click "Deploy"

4. **Auto-deployment enabled**:
   - Every push to `main` = automatic deployment
   - Pull requests get preview deployments

---

### Option C: Vercel Dashboard (No CLI)

**Why**: No command line needed

**Steps**:

1. Build project locally:
   ```bash
   cd /Users/paulolusola/PyCharmMiscProject/smart-inventory-manager/frontend
   npm install
   npm run build
   ```

2. Go to [vercel.com/new](https://vercel.com/new)

3. Click "Continue with Email" or GitHub

4. Drag and drop the `frontend` folder

5. Click "Deploy"

6. Wait 1-2 minutes

7. **Done!** Vercel provides live URL

---

## Post-Deployment Configuration

### Environment Variables (If Needed)

In Vercel Dashboard:

1. Go to your project
2. Settings → Environment Variables
3. Add:
   - Key: `NEXT_PUBLIC_API_URL`
   - Value: `https://pauloski07-sales-inventory-forecasting.hf.space`
4. Save
5. Redeploy

**Note**: This is already set in `.env.local`, so usually not needed!

### Custom Domain

1. **In Vercel Dashboard**:
   - Project → Settings → Domains
   - Click "Add"
   - Enter your domain (e.g., `inventory.yourdomain.com`)

2. **Update DNS** (at your domain registrar):
   - Add CNAME record:
     - Name: `inventory` (or `@` for root)
     - Value: `cname.vercel-dns.com`
   - Or use A records Vercel provides

3. **Wait for DNS propagation** (5 minutes - 48 hours)

4. **SSL Certificate**: Auto-provisioned by Vercel (free!)

---

## Verification Steps

### Test Your Deployment

1. **Homepage**: Should redirect to `/dashboard`
   ```
   https://your-app.vercel.app
   ```

2. **Dashboard**: Should show KPIs
   ```
   https://your-app.vercel.app/dashboard
   ```

3. **Sales Trends**: Should show charts
   ```
   https://your-app.vercel.app/trends
   ```

4. **API Connection**: Check browser console for errors
   - Open DevTools (F12)
   - Go to Network tab
   - Should see successful requests to HF Spaces API

### Health Checks

**Backend API**:
```bash
curl https://pauloski07-sales-inventory-forecasting.hf.space/health
```

Expected: `{"status":"healthy",...}`

**Frontend**:
```bash
curl https://your-app.vercel.app
```

Expected: HTML response

---

## Common Issues & Solutions

### Issue 1: Build Fails

**Error**: `Module not found`

**Solution**:
```bash
rm -rf node_modules package-lock.json
npm install --legacy-peer-deps
npm run build
```

### Issue 2: API Not Loading

**Error**: CORS or network errors in browser console

**Solutions**:
1. Check backend is live: https://pauloski07-sales-inventory-forecasting.hf.space/health
2. Verify `.env.local` has correct API URL
3. Clear browser cache and retry

### Issue 3: Blank Page

**Error**: White screen, no content

**Solutions**:
1. Check browser console for JavaScript errors
2. Ensure build was successful
3. Try hard refresh: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)

### Issue 4: Slow Load Times

**Cause**: Hugging Face Spaces sleeps after inactivity

**Solution**: First load may take 30-60 seconds (waking up API)
- Subsequent loads are fast
- Upgrade HF Spaces to persistent hardware if needed

---

## Monitoring & Analytics

### Vercel Analytics (Free)

1. Go to project dashboard
2. Click "Analytics" tab
3. View:
   - Page views
   - Load times
   - Errors
   - Geographic distribution

### Vercel Logs

1. Project dashboard → "Logs"
2. See real-time server logs
3. Debug issues

---

## Maintenance

### Update Deployment

**If using CLI**:
```bash
cd frontend
# Make changes...
vercel --prod
```

**If using GitHub integration**:
```bash
git add .
git commit -m "Update: ..."
git push
```

Auto-deploys in ~2 minutes!

### Rollback

In Vercel dashboard:
1. Go to "Deployments"
2. Find previous successful deployment
3. Click "..." → "Promote to Production"

---

## Architecture

```
User Browser
     ↓
Vercel (Frontend)
     ↓ API Calls
Hugging Face Spaces (Backend)
     ↓
SQLite Database
```

### Flow:
1. User visits `your-app.vercel.app`
2. Vercel serves Next.js frontend
3. Frontend calls HF Spaces API
4. HF Spaces processes request, queries database
5. Returns JSON data
6. Frontend displays charts/tables

---

## Performance Optimization

### Built-in (Already Configured):
- ✅ Static Site Generation (SSG)
- ✅ API response caching (60s)
- ✅ Image optimization
- ✅ Code splitting
- ✅ Compression

### Optional Improvements:
1. **Add loading skeletons** for better perceived performance
2. **Implement pagination** for large tables
3. **Add React Query** for advanced caching
4. **Use Incremental Static Regeneration (ISR)** for some pages

---

## Cost

### Free Tier (Perfect for This Project):
- **Vercel**:
  - ✅ Unlimited deployments
  - ✅ 100 GB bandwidth/month
  - ✅ Automatic SSL
  - ✅ Perfect for personal projects

- **Hugging Face Spaces**:
  - ✅ Free tier (already using)
  - ⚠️ May sleep after inactivity
  - 💰 Upgrade to persistent for $9/month (optional)

### Paid Options (If Needed):
- **Vercel Pro**: $20/month (more bandwidth, analytics)
- **Custom domain**: ~$12/year (from registrar)

**Total**: $0/month for free tier, works perfectly!

---

## Security Checklist

- [ ] Environment variables not exposed in client code
- [ ] API URL uses HTTPS
- [ ] No sensitive data in repository
- [ ] CORS properly configured on backend
- [ ] For production: Consider adding authentication

---

## 🎉 Success Criteria

Your deployment is successful if:

- ✅ App loads at Vercel URL
- ✅ Dashboard shows live data
- ✅ Charts render properly
- ✅ All pages are accessible
- ✅ No console errors
- ✅ Mobile responsive
- ✅ Fast load times (<3 seconds)

---

## Next Steps After Deployment

1. **Share your live URL**!
2. **Test all features**
3. **Get feedback**
4. **Expand stub pages** (patterns, reports, etc.)
5. **Add custom branding**
6. **Consider custom domain**

---

## Support

- **Vercel Docs**: https://vercel.com/docs
- **Next.js Docs**: https://nextjs.org/docs
- **Issue**: Check backend health first
- **Help**: Vercel has excellent support community

---

**Your URLs:**

- Frontend (Vercel): `https://your-project-name.vercel.app` ⬅️ Deploy here!
- Backend (HF Spaces): `https://pauloski07-sales-inventory-forecasting.hf.space` ✅ Already live!

**Estimated Deployment Time**: 2-5 minutes

**Difficulty**: Easy! 🟢

Good luck! 🚀
