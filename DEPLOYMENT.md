# Deployment Guide

This guide covers deploying the Fresh or Rotten web application to Railway or Render.

## Prerequisites

- Trained model file: `models/deep_efficientnet_b0.pth`
- Git repository initialized
- GitHub account (for Railway/Render)

## Option 1: Deploy to Railway

Railway is recommended for its simplicity and generous free tier.

### Steps:

1. **Install Railway CLI** (optional but helpful):
   ```bash
   npm install -g @railway/cli
   ```

2. **Login to Railway**:
   ```bash
   railway login
   ```

3. **Initialize project**:
   ```bash
   railway init
   ```

4. **Deploy**:
   ```bash
   railway up
   ```

5. **Get public URL**:
   ```bash
   railway open
   ```

### Alternative: Deploy via GitHub

1. Go to [railway.app](https://railway.app)
2. Click "Start a New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway will auto-detect the `Procfile` and deploy
5. Add custom domain or use Railway's provided URL

### Environment Variables (if needed):

Railway auto-detects most settings. If needed, set:
- `PORT`: Auto-set by Railway
- `PYTHON_VERSION`: 3.11.0 (from runtime.txt)

## Option 2: Deploy to Render

Render offers similar functionality with a web-based deployment flow.

### Steps:

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Deploy to Render"
   git push origin main
   ```

2. **Create Render account**: Go to [render.com](https://render.com)

3. **Create new Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the DL-CV repository

4. **Configure build settings**:
   - **Name**: fresh-or-rotten
   - **Region**: Choose closest to you
   - **Branch**: main
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 "app:create_app()"`

5. **Advanced settings**:
   - **Instance Type**: Free (or Starter for better performance)
   - **Environment Variables**: None needed

6. **Deploy**: Click "Create Web Service"

7. **Wait for build**: First deployment takes 5-10 minutes

8. **Access your app**: Render provides a URL like `https://fresh-or-rotten.onrender.com`

## Important Notes

### Model File Size

The EfficientNet-B0 model (~20MB) should be committed to your repository:

```bash
git lfs install  # If using Git LFS (recommended for large files)
git add models/deep_efficientnet_b0.pth
git commit -m "Add trained model"
git push
```

**Alternative**: If model is too large, host it separately (e.g., Google Drive, AWS S3) and download it during deployment.

### Memory Considerations

- **Railway Free Tier**: 512MB RAM - may be tight with PyTorch
- **Render Free Tier**: 512MB RAM - same concern
- **Upgrade if needed**: ~$7/month for 1GB RAM

If you hit memory limits:
1. Use model quantization
2. Use a smaller model backbone
3. Upgrade to paid tier

### Cold Starts

Free tiers may have "cold starts" (app sleeps after inactivity):
- First request after sleep may take 10-30 seconds
- Keep-alive services can ping your app to prevent sleep

## Testing Deployment

After deployment:

1. **Check health endpoint**:
   ```bash
   curl https://your-app-url.com/health
   ```

2. **Test prediction API**:
   ```bash
   curl -X POST -F "image=@test_image.jpg" https://your-app-url.com/api/predict
   ```

3. **Test camera access**: Open app on your phone (must use HTTPS for camera!)

## Troubleshooting

### Build fails:

- Check logs for missing dependencies
- Ensure `requirements.txt` is complete
- Verify Python version in `runtime.txt`

### App crashes on startup:

- Check that model file exists in `models/`
- Verify model path in config
- Check logs for import errors

### Out of memory:

- Reduce batch size in inference
- Use model quantization
- Upgrade to larger instance

### Camera doesn't work:

- Ensure app is served over HTTPS (Railway/Render provide this)
- Check browser permissions
- Test on multiple devices

## Custom Domain (Optional)

Both Railway and Render support custom domains:

1. **Railway**:
   - Project Settings → Domains → Add Custom Domain
   - Update DNS records as instructed

2. **Render**:
   - Dashboard → Custom Domains → Add Domain
   - Update DNS CNAME record

## Monitoring

### Railway:
- View logs: `railway logs`
- Metrics: Dashboard shows CPU/Memory usage

### Render:
- Logs available in web dashboard
- Metrics tab shows resource usage

## Cost Estimates

### Free Tier (Both):
- ✅ Sufficient for development/testing
- ✅ Works for low-traffic demos
- ⚠️ May have cold starts
- ⚠️ 512MB RAM (tight for PyTorch)

### Starter Tier (~$7-10/month):
- ✅ 1-2GB RAM
- ✅ No cold starts
- ✅ Better performance
- ✅ Suitable for production demos

## Next Steps

After successful deployment:

1. ✅ Test on multiple devices (phone, tablet, laptop)
2. ✅ Update README.md with live demo URL
3. ✅ Share link with team/graders
4. ✅ Monitor for 24h to ensure stability
5. ✅ Set up uptime monitoring (e.g., UptimeRobot)

## Support

- Railway: [docs.railway.app](https://docs.railway.app)
- Render: [render.com/docs](https://render.com/docs)
- Issues: Check app logs first, then platform documentation

---

**Good luck with your deployment! 🚀**
