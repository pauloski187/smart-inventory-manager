# ⚡ Quick Start - 5 Minutes to Deployed

## 🎯 Goal

Get your inventory dashboard live on the internet in 5 minutes!

## ✅ Prerequisites Check

- [ ] You have a computer with terminal access
- [ ] Node.js is installed (`node --version` should show v18+)
- [ ] You can access https://pauloski07-sales-inventory-forecasting.hf.space/health (should return JSON)

All good? Let's go! 🚀

---

## Step 1: Install Dependencies (1 minute)

```bash
cd /Users/paulolusola/PyCharmMiscProject/smart-inventory-manager/frontend
npm install
```

**Wait for**: "added XXX packages"

---

## Step 2: Test Locally (1 minute)

```bash
npm run dev
```

**Open**: http://localhost:3000

**You should see**: Dashboard with charts and data!

**If it works**: Press `Ctrl+C` to stop. Move to Step 3!

**If it doesn't work**:
- Check backend is up: https://pauloski07-sales-inventory-forecasting.hf.space/health
- Try: `rm -rf node_modules && npm install --legacy-peer-deps && npm run dev`

---

## Step 3: Deploy to Vercel (3 minutes)

### Option A: One Command Deploy

```bash
npx vercel --prod
```

**Follow prompts**:
1. "Set up and deploy?" → **Yes**
2. "Which scope?" → Choose your account (or sign up)
3. "Link to existing project?" → **No**
4. "Project name?" → `smart-inventory` (or your choice)
5. "Directory?" → **./
**
6. "Want to override settings?" → **No**

**Result**:
```
✅ Production: https://smart-inventory-xxx.vercel.app
```

**Test your live URL!** 🎉

### Option B: GitHub Then Vercel

**If you prefer continuous deployment**:

```bash
# Initialize git
git init
git add .
git commit -m "Initial commit"

# Create repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git push -u origin main

# Go to vercel.com/new
# Import your GitHub repo
# Click Deploy
```

---

## ✅ Verification

Visit your Vercel URL and check:

- [ ] Dashboard loads
- [ ] KPI cards show numbers
- [ ] Sidebar navigation works
- [ ] Sales Trends page shows charts
- [ ] Categories page shows pie chart
- [ ] Products page shows table
- [ ] Alerts page shows inventory alerts

**All working?** Congratulations! 🎊

---

## 🎨 What You Have Now

- ✅ Professional dashboard UI
- ✅ Live data from your API
- ✅ Multiple analytics pages
- ✅ Responsive design (works on mobile)
- ✅ Fast performance
- ✅ Free hosting
- ✅ Automatic SSL
- ✅ Custom URL from Vercel

---

## 🔧 Customize (Optional)

### Change App Name

Edit `src/components/layout/Sidebar.tsx`:
```tsx
<h1 className="text-xl font-bold text-primary">
  Your Company Name  {/* Change this */}
</h1>
```

Redeploy:
```bash
vercel --prod
```

### Add Custom Domain

1. Buy domain (namecheap.com, etc.)
2. In Vercel dashboard → Settings → Domains
3. Add your domain
4. Follow DNS instructions
5. Wait ~10 minutes
6. Done! Your app at `yourdomain.com`

---

## 🚨 Troubleshooting

### "Module not found" error

```bash
npm install --legacy-peer-deps
```

### Build fails

```bash
npm run build
```

Fix any errors shown, then redeploy.

### API not loading

Check: https://pauloski07-sales-inventory-forecasting.hf.space/health

If down, wait a minute (HF Spaces may be waking up).

### Deployment fails

```bash
vercel --prod --force
```

---

## 📚 Next Steps

1. ✅ **Test all pages**: Click through everything
2. 📱 **Test on mobile**: Open on your phone
3. 🎨 **Customize branding**: Change colors, logo
4. 📊 **Expand stub pages**: Complete patterns, reports, etc.
5. 🔗 **Share your URL**: Show it off!

---

## 🎯 Your Live URLs

**Frontend (Your New Dashboard)**:
```
https://your-project-name.vercel.app
```

**Backend (Already Live)**:
```
https://pauloski07-sales-inventory-forecasting.hf.space
```

**API Docs**:
```
https://pauloski07-sales-inventory-forecasting.hf.space/docs
```

---

## 💡 Pro Tips

1. **Bookmark your Vercel URL**
2. **Share with others** to get feedback
3. **Check Vercel dashboard** for analytics
4. **Make changes locally**, then `vercel --prod` to update
5. **Use GitHub** for version control and automatic deployments

---

## 🎉 That's It!

You now have a **production-ready inventory management dashboard** deployed to the internet!

**Time taken**: ~5 minutes ⏱️
**Cost**: $0 💰
**Difficulty**: Easy 🟢

Congratulations! 🚀

---

**Need help?** Check:
- README.md (detailed documentation)
- DEPLOYMENT.md (troubleshooting guide)
- Backend API health: https://pauloski07-sales-inventory-forecasting.hf.space/health
