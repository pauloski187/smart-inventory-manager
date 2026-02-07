/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'https://pauloski07-sales-inventory-forecasting.hf.space',
  },
}

module.exports = nextConfig
