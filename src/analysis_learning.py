#!/usr/bin/env python3
"""
Analysis Learning Module for DataBot AI

This module implements a learning system that improves data analysis capabilities
by learning from successful analysis patterns, insights, and outcomes.
It focuses solely on data analysis improvement, not human prompt learning.
"""

import os
import json
import time
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import pickle

@dataclass
class AnalysisPattern:
    """Represents a successful data analysis pattern"""
    pattern_id: str
    data_characteristics: Dict[str, Any]  # Data type, size, structure, etc.
    analysis_method: str  # Statistical method, visualization type, etc.
    insight_quality: float  # 0-1 score of insight quality
    execution_time: float  # Time taken for analysis
    success_metrics: Dict[str, float]  # Various success indicators
    timestamp: str
    context: Dict[str, Any]  # Additional context about the analysis

@dataclass
class InsightRecord:
    """Records insights generated and their effectiveness"""
    insight_id: str
    dataset_id: str
    insight_text: str
    insight_type: str  # trend, correlation, anomaly, prediction, etc.
    confidence_score: float
    validation_score: Optional[float]  # If validated later
    impact_score: Optional[float]  # Business/analytical impact
    timestamp: str
    analysis_method: str
    data_features: Dict[str, Any]

@dataclass
class AnalysisOutcome:
    """Records the outcome and effectiveness of an analysis"""
    outcome_id: str
    analysis_id: str
    success_indicators: Dict[str, float]
    user_satisfaction: Optional[float]
    insight_accuracy: Optional[float]
    computational_efficiency: float
    timestamp: str

class AnalysisLearningSystem:
    """
    Learning system that improves data analysis capabilities over time
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the analysis learning system
        
        Args:
            data_dir: Directory for storing learning data
        """
        self.data_dir = data_dir
        self.learning_dir = os.path.join(data_dir, "learning")
        os.makedirs(self.learning_dir, exist_ok=True)
        
        # Learning data storage
        self.patterns_file = os.path.join(self.learning_dir, "analysis_patterns.json")
        self.insights_file = os.path.join(self.learning_dir, "insights_record.json")
        self.outcomes_file = os.path.join(self.learning_dir, "analysis_outcomes.json")
        self.knowledge_graph_file = os.path.join(self.learning_dir, "knowledge_graph.pkl")
        
        # In-memory caches
        self.patterns: List[AnalysisPattern] = []
        self.insights: List[InsightRecord] = []
        self.outcomes: List[AnalysisOutcome] = []
        self.knowledge_graph: Dict[str, Any] = {}
        
        # Learning metrics
        self.method_effectiveness: Dict[str, List[float]] = defaultdict(list)
        self.data_type_preferences: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.insight_type_success: Dict[str, List[float]] = defaultdict(list)
        
        # Load existing learning data
        self._load_learning_data()
    
    def _load_learning_data(self):
        """Load existing learning data from files"""
        try:
            # Load patterns
            if os.path.exists(self.patterns_file):
                with open(self.patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                    self.patterns = [AnalysisPattern(**p) for p in patterns_data]
            
            # Load insights
            if os.path.exists(self.insights_file):
                with open(self.insights_file, 'r') as f:
                    insights_data = json.load(f)
                    self.insights = [InsightRecord(**i) for i in insights_data]
            
            # Load outcomes
            if os.path.exists(self.outcomes_file):
                with open(self.outcomes_file, 'r') as f:
                    outcomes_data = json.load(f)
                    self.outcomes = [AnalysisOutcome(**o) for o in outcomes_data]
            
            # Load knowledge graph
            if os.path.exists(self.knowledge_graph_file):
                with open(self.knowledge_graph_file, 'rb') as f:
                    self.knowledge_graph = pickle.load(f)
            
            # Rebuild learning metrics
            self._rebuild_learning_metrics()
            
            print(f"Loaded {len(self.patterns)} patterns, {len(self.insights)} insights, {len(self.outcomes)} outcomes")
            
        except Exception as e:
            print(f"Error loading learning data: {e}")
    
    def _save_learning_data(self):
        """Save learning data to files"""
        try:
            # Save patterns
            with open(self.patterns_file, 'w') as f:
                json.dump([asdict(p) for p in self.patterns], f, indent=2)
            
            # Save insights
            with open(self.insights_file, 'w') as f:
                json.dump([asdict(i) for i in self.insights], f, indent=2)
            
            # Save outcomes
            with open(self.outcomes_file, 'w') as f:
                json.dump([asdict(o) for o in self.outcomes], f, indent=2)
            
            # Save knowledge graph
            with open(self.knowledge_graph_file, 'wb') as f:
                pickle.dump(self.knowledge_graph, f)
                
        except Exception as e:
            print(f"Error saving learning data: {e}")
    
    def _rebuild_learning_metrics(self):
        """Rebuild learning metrics from stored data"""
        self.method_effectiveness.clear()
        self.data_type_preferences.clear()
        self.insight_type_success.clear()
        
        # Rebuild from patterns
        for pattern in self.patterns:
            method = pattern.analysis_method
            self.method_effectiveness[method].append(pattern.insight_quality)
            
            # Data type preferences
            for data_type, characteristics in pattern.data_characteristics.items():
                if data_type not in self.data_type_preferences:
                    self.data_type_preferences[data_type] = {}
                self.data_type_preferences[data_type][method] = pattern.insight_quality
        
        # Rebuild from insights
        for insight in self.insights:
            insight_type = insight.insight_type
            if insight.validation_score is not None:
                self.insight_type_success[insight_type].append(insight.validation_score)
    
    def record_analysis_pattern(self, 
                              data_characteristics: Dict[str, Any],
                              analysis_method: str,
                              insight_quality: float,
                              execution_time: float,
                              success_metrics: Dict[str, float],
                              context: Dict[str, Any] = None) -> str:
        """
        Record a successful analysis pattern for learning
        
        Args:
            data_characteristics: Characteristics of the analyzed data
            analysis_method: Method used for analysis
            insight_quality: Quality score of insights generated (0-1)
            execution_time: Time taken for analysis
            success_metrics: Various success indicators
            context: Additional context
            
        Returns:
            Pattern ID
        """
        pattern_id = hashlib.md5(
            f"{analysis_method}_{time.time()}".encode()
        ).hexdigest()[:12]
        
        pattern = AnalysisPattern(
            pattern_id=pattern_id,
            data_characteristics=data_characteristics,
            analysis_method=analysis_method,
            insight_quality=insight_quality,
            execution_time=execution_time,
            success_metrics=success_metrics,
            timestamp=datetime.now().isoformat(),
            context=context or {}
        )
        
        self.patterns.append(pattern)
        
        # Update learning metrics
        self.method_effectiveness[analysis_method].append(insight_quality)
        
        # Update data type preferences
        for data_type in data_characteristics.keys():
            if data_type not in self.data_type_preferences:
                self.data_type_preferences[data_type] = {}
            self.data_type_preferences[data_type][analysis_method] = insight_quality
        
        self._save_learning_data()
        return pattern_id
    
    def record_insight(self,
                      dataset_id: str,
                      insight_text: str,
                      insight_type: str,
                      confidence_score: float,
                      analysis_method: str,
                      data_features: Dict[str, Any]) -> str:
        """
        Record an insight generated during analysis
        
        Args:
            dataset_id: ID of the dataset analyzed
            insight_text: The insight generated
            insight_type: Type of insight (trend, correlation, etc.)
            confidence_score: Confidence in the insight
            analysis_method: Method used to generate insight
            data_features: Features of the data that led to insight
            
        Returns:
            Insight ID
        """
        insight_id = hashlib.md5(
            f"{dataset_id}_{insight_text}_{time.time()}".encode()
        ).hexdigest()[:12]
        
        insight = InsightRecord(
            insight_id=insight_id,
            dataset_id=dataset_id,
            insight_text=insight_text,
            insight_type=insight_type,
            confidence_score=confidence_score,
            validation_score=None,
            impact_score=None,
            timestamp=datetime.now().isoformat(),
            analysis_method=analysis_method,
            data_features=data_features
        )
        
        self.insights.append(insight)
        self._save_learning_data()
        return insight_id
    
    def validate_insight(self, insight_id: str, validation_score: float, impact_score: float = None):
        """
        Validate an insight with actual outcomes
        
        Args:
            insight_id: ID of the insight to validate
            validation_score: How accurate the insight was (0-1)
            impact_score: Business/analytical impact (0-1)
        """
        for insight in self.insights:
            if insight.insight_id == insight_id:
                insight.validation_score = validation_score
                if impact_score is not None:
                    insight.impact_score = impact_score
                
                # Update learning metrics
                self.insight_type_success[insight.insight_type].append(validation_score)
                break
        
        self._save_learning_data()
    
    def get_recommended_method(self, data_characteristics: Dict[str, Any]) -> Tuple[str, float]:
        """
        Get recommended analysis method based on learned patterns
        
        Args:
            data_characteristics: Characteristics of the data to analyze
            
        Returns:
            Tuple of (recommended_method, confidence_score)
        """
        method_scores = defaultdict(list)
        
        # Score methods based on similar data characteristics
        for pattern in self.patterns:
            similarity = self._calculate_data_similarity(
                data_characteristics, 
                pattern.data_characteristics
            )
            
            if similarity > 0.3:  # Threshold for similarity
                weighted_score = pattern.insight_quality * similarity
                method_scores[pattern.analysis_method].append(weighted_score)
        
        if not method_scores:
            return "exploratory_analysis", 0.5  # Default fallback
        
        # Calculate average scores for each method
        method_averages = {
            method: np.mean(scores) 
            for method, scores in method_scores.items()
        }
        
        best_method = max(method_averages, key=method_averages.get)
        confidence = method_averages[best_method]
        
        return best_method, confidence
    
    def _calculate_data_similarity(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two data characteristic dictionaries
        
        Args:
            data1: First data characteristics
            data2: Second data characteristics
            
        Returns:
            Similarity score (0-1)
        """
        common_keys = set(data1.keys()) & set(data2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = data1[key], data2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                if val1 == 0 and val2 == 0:
                    sim = 1.0
                else:
                    sim = 1.0 - abs(val1 - val2) / max(abs(val1), abs(val2), 1)
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity (simple)
                sim = 1.0 if val1 == val2 else 0.0
            else:
                # Default similarity
                sim = 1.0 if val1 == val2 else 0.0
            
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """
        Get insights about what the system has learned
        
        Returns:
            Dictionary of learning insights
        """
        insights = {
            "total_patterns": len(self.patterns),
            "total_insights": len(self.insights),
            "total_outcomes": len(self.outcomes),
            "method_effectiveness": {},
            "insight_type_success": {},
            "data_type_preferences": dict(self.data_type_preferences),
            "learning_trends": self._calculate_learning_trends()
        }
        
        # Calculate method effectiveness
        for method, scores in self.method_effectiveness.items():
            insights["method_effectiveness"][method] = {
                "average_quality": np.mean(scores),
                "count": len(scores),
                "improvement_trend": self._calculate_trend(scores)
            }
        
        # Calculate insight type success
        for insight_type, scores in self.insight_type_success.items():
            insights["insight_type_success"][insight_type] = {
                "average_validation": np.mean(scores),
                "count": len(scores),
                "improvement_trend": self._calculate_trend(scores)
            }
        
        return insights
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate trend in scores (improving, declining, stable)"""
        if len(scores) < 3:
            return "insufficient_data"
        
        recent = np.mean(scores[-3:])
        older = np.mean(scores[:-3])
        
        if recent > older + 0.05:
            return "improving"
        elif recent < older - 0.05:
            return "declining"
        else:
            return "stable"
    
    def _calculate_learning_trends(self) -> Dict[str, Any]:
        """Calculate overall learning trends"""
        if len(self.patterns) < 5:
            return {"status": "insufficient_data"}
        
        # Group patterns by time periods
        recent_patterns = [p for p in self.patterns[-10:]]
        older_patterns = [p for p in self.patterns[:-10]] if len(self.patterns) > 10 else []
        
        if not older_patterns:
            return {"status": "early_learning_phase"}
        
        recent_quality = np.mean([p.insight_quality for p in recent_patterns])
        older_quality = np.mean([p.insight_quality for p in older_patterns])
        
        recent_efficiency = np.mean([p.execution_time for p in recent_patterns])
        older_efficiency = np.mean([p.execution_time for p in older_patterns])
        
        return {
            "status": "active_learning",
            "quality_improvement": recent_quality - older_quality,
            "efficiency_improvement": older_efficiency - recent_efficiency,  # Lower is better
            "learning_velocity": len(recent_patterns) / max(len(older_patterns), 1)
        }
    
    def optimize_for_dataset(self, dataset_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get optimization recommendations for a specific dataset
        
        Args:
            dataset_characteristics: Characteristics of the dataset
            
        Returns:
            Optimization recommendations
        """
        recommendations = {
            "recommended_methods": [],
            "expected_insights": [],
            "optimization_tips": [],
            "confidence_level": 0.0
        }
        
        # Find similar datasets and successful patterns
        similar_patterns = []
        for pattern in self.patterns:
            similarity = self._calculate_data_similarity(
                dataset_characteristics, 
                pattern.data_characteristics
            )
            if similarity > 0.4:
                similar_patterns.append((pattern, similarity))
        
        if not similar_patterns:
            recommendations["confidence_level"] = 0.2
            recommendations["optimization_tips"].append("No similar datasets found in learning history")
            return recommendations
        
        # Sort by similarity and quality
        similar_patterns.sort(key=lambda x: x[1] * x[0].insight_quality, reverse=True)
        
        # Get top recommendations
        top_patterns = similar_patterns[:5]
        
        method_scores = defaultdict(list)
        for pattern, similarity in top_patterns:
            weighted_score = pattern.insight_quality * similarity
            method_scores[pattern.analysis_method].append(weighted_score)
        
        # Recommend top methods
        for method, scores in sorted(method_scores.items(), 
                                   key=lambda x: np.mean(x[1]), reverse=True)[:3]:
            recommendations["recommended_methods"].append({
                "method": method,
                "expected_quality": np.mean(scores),
                "confidence": min(len(scores) / 5.0, 1.0)
            })
        
        # Predict likely insights
        insight_types = Counter()
        for insight in self.insights:
            for pattern, similarity in top_patterns:
                if insight.analysis_method == pattern.analysis_method:
                    insight_types[insight.insight_type] += similarity
        
        for insight_type, score in insight_types.most_common(3):
            avg_validation = np.mean([
                i.validation_score for i in self.insights 
                if i.insight_type == insight_type and i.validation_score is not None
            ]) if any(i.validation_score for i in self.insights if i.insight_type == insight_type) else 0.5
            
            recommendations["expected_insights"].append({
                "type": insight_type,
                "likelihood": min(score, 1.0),
                "expected_accuracy": avg_validation
            })
        
        recommendations["confidence_level"] = min(len(similar_patterns) / 10.0, 1.0)
        
        return recommendations