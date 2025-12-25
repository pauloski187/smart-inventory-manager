"""
ABC Analysis Module

Classifies products by revenue contribution using the Pareto principle.
"""

import pandas as pd
from typing import List, Dict
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
import logging

from ..models import Order, Product

logger = logging.getLogger(__name__)


class ABCAnalyzer:
    """
    ABC Analysis for product classification.

    Classifies products into:
    - A Items (Top 20%): ~80% of revenue - Monitor closely
    - B Items (Next 30%): ~15% of revenue - Moderate attention
    - C Items (Bottom 50%): ~5% of revenue - Minimal monitoring
    """

    def __init__(self, db: Session):
        self.db = db

    def calculate_product_revenue(self) -> pd.DataFrame:
        """
        Calculate total revenue per product.

        Returns:
            DataFrame with product_id, product_name, category, total_revenue, total_quantity
        """
        # Query revenue per product
        results = self.db.query(
            Order.product_id,
            func.sum(Order.total_amount).label('total_revenue'),
            func.sum(Order.quantity).label('total_quantity')
        ).filter(
            Order.order_status == 'Delivered'
        ).group_by(
            Order.product_id
        ).all()

        if not results:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(results, columns=['product_id', 'total_revenue', 'total_quantity'])

        # Add product details
        products = {p.id: p for p in self.db.query(Product).all()}
        df['product_name'] = df['product_id'].map(lambda x: products.get(x, Product()).name)
        df['category'] = df['product_id'].map(lambda x: products.get(x, Product()).category)

        # Sort by revenue descending
        df = df.sort_values('total_revenue', ascending=False)

        return df

    def perform_analysis(self) -> pd.DataFrame:
        """
        Perform ABC analysis on all products.

        Returns:
            DataFrame with ABC classification for each product
        """
        df = self.calculate_product_revenue()

        if df.empty:
            return df

        # Calculate cumulative revenue percentage
        total_revenue = df['total_revenue'].sum()
        df['revenue_pct'] = (df['total_revenue'] / total_revenue * 100).round(2)
        df['cumulative_pct'] = df['revenue_pct'].cumsum().round(2)

        # Classify products
        def classify(cumulative_pct):
            if cumulative_pct <= 80:
                return 'A'
            elif cumulative_pct <= 95:
                return 'B'
            else:
                return 'C'

        df['abc_class'] = df['cumulative_pct'].apply(classify)

        # Calculate rank
        df['rank'] = range(1, len(df) + 1)

        return df[['product_id', 'product_name', 'category', 'total_revenue',
                   'total_quantity', 'revenue_pct', 'cumulative_pct', 'abc_class', 'rank']]

    def get_class_summary(self) -> Dict:
        """
        Get summary statistics for each ABC class.

        Returns:
            Dictionary with statistics per class
        """
        df = self.perform_analysis()

        if df.empty:
            return {}

        summary = {}
        for abc_class in ['A', 'B', 'C']:
            class_df = df[df['abc_class'] == abc_class]
            summary[abc_class] = {
                'product_count': len(class_df),
                'product_pct': round(len(class_df) / len(df) * 100, 1),
                'total_revenue': round(class_df['total_revenue'].sum(), 2),
                'revenue_pct': round(class_df['revenue_pct'].sum(), 1),
                'total_quantity': int(class_df['total_quantity'].sum()),
                'avg_revenue_per_product': round(class_df['total_revenue'].mean(), 2)
            }

        return summary

    def get_recommendations(self) -> List[Dict]:
        """
        Get management recommendations based on ABC analysis.

        Returns:
            List of recommendations per class
        """
        summary = self.get_class_summary()

        if not summary:
            return []

        recommendations = []

        if 'A' in summary:
            recommendations.append({
                'class': 'A',
                'title': 'High-Value Products (A Items)',
                'product_count': summary['A']['product_count'],
                'revenue_contribution': f"{summary['A']['revenue_pct']}%",
                'recommendations': [
                    'Maintain tight inventory control',
                    'Monitor stock levels daily',
                    'Prioritize these items for restocking',
                    'Negotiate better supplier terms',
                    'Implement safety stock policies'
                ]
            })

        if 'B' in summary:
            recommendations.append({
                'class': 'B',
                'title': 'Medium-Value Products (B Items)',
                'product_count': summary['B']['product_count'],
                'revenue_contribution': f"{summary['B']['revenue_pct']}%",
                'recommendations': [
                    'Review inventory weekly',
                    'Standard safety stock levels',
                    'Monitor for movement to A or C class',
                    'Consider promotional opportunities'
                ]
            })

        if 'C' in summary:
            recommendations.append({
                'class': 'C',
                'title': 'Low-Value Products (C Items)',
                'product_count': summary['C']['product_count'],
                'revenue_contribution': f"{summary['C']['revenue_pct']}%",
                'recommendations': [
                    'Minimize inventory investment',
                    'Consider reducing SKU count',
                    'Bulk order to reduce costs',
                    'Review for potential discontinuation',
                    'May include dead stock candidates'
                ]
            })

        return recommendations

    def get_top_products(
        self,
        n: int = 10,
        abc_class: str = None
    ) -> List[Dict]:
        """
        Get top N products by revenue.

        Args:
            n: Number of products to return
            abc_class: Optional filter by ABC class

        Returns:
            List of top products with details
        """
        df = self.perform_analysis()

        if df.empty:
            return []

        if abc_class:
            df = df[df['abc_class'] == abc_class]

        top_products = df.head(n)

        return top_products.to_dict('records')

    def get_category_breakdown(self) -> pd.DataFrame:
        """
        Get ABC analysis breakdown by category.

        Returns:
            DataFrame with category-level ABC distribution
        """
        df = self.perform_analysis()

        if df.empty:
            return pd.DataFrame()

        # Group by category and ABC class
        breakdown = df.groupby(['category', 'abc_class']).agg({
            'product_id': 'count',
            'total_revenue': 'sum',
            'total_quantity': 'sum'
        }).reset_index()

        breakdown.columns = ['category', 'abc_class', 'product_count', 'total_revenue', 'total_quantity']

        # Pivot for easier reading
        pivot = breakdown.pivot_table(
            index='category',
            columns='abc_class',
            values=['product_count', 'total_revenue'],
            fill_value=0
        )

        return breakdown

    def to_dict_list(self) -> List[Dict]:
        """
        Convert analysis results to list of dictionaries for API response.
        """
        df = self.perform_analysis()
        if df.empty:
            return []

        return df.to_dict('records')
