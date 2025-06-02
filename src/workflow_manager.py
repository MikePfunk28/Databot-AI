#!/usr/bin/env python3
"""
Workflow Manager for DataBot AI System

This module provides workflow orchestration capabilities for data analysis tasks,
with optional integration with Microsoft PromptFlow.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Types of tasks in the workflow"""
    DATA_INGESTION = "data_ingestion"
    EMBEDDING_GENERATION = "embedding_generation"
    DATA_ANALYSIS = "data_analysis"
    INSIGHT_GENERATION = "insight_generation"
    VISUALIZATION = "visualization"
    REPORT_GENERATION = "report_generation"
    CUSTOM = "custom"


@dataclass
class WorkflowTask:
    """Individual task in a workflow"""
    id: str
    name: str
    task_type: TaskType
    dependencies: List[str]
    parameters: Dict[str, Any]
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['task_type'] = self.task_type.value
        data['status'] = self.status.value
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowTask':
        """Create from dictionary"""
        task = cls(
            id=data['id'],
            name=data['name'],
            task_type=TaskType(data['task_type']),
            dependencies=data['dependencies'],
            parameters=data['parameters'],
            status=WorkflowStatus(data['status']),
            result=data.get('result'),
            error=data.get('error')
        )
        if data.get('start_time'):
            task.start_time = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            task.end_time = datetime.fromisoformat(data['end_time'])
        return task


@dataclass
class Workflow:
    """Complete workflow definition"""
    id: str
    name: str
    description: str
    tasks: List[WorkflowTask]
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'tasks': [task.to_dict() for task in self.tasks],
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Workflow':
        """Create from dictionary"""
        workflow = cls(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            tasks=[WorkflowTask.from_dict(task_data) for task_data in data['tasks']],
            status=WorkflowStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at'])
        )
        if data.get('started_at'):
            workflow.started_at = datetime.fromisoformat(data['started_at'])
        if data.get('completed_at'):
            workflow.completed_at = datetime.fromisoformat(data['completed_at'])
        return workflow


class WorkflowManager:
    """
    Manages workflow execution for data analysis tasks
    """
    
    def __init__(self, data_dir: str, components: Dict[str, Any]):
        """
        Initialize workflow manager
        
        Args:
            data_dir: Directory for storing workflow data
            components: Dictionary of system components (model_wrapper, data_ingestion, etc.)
        """
        self.data_dir = data_dir
        self.workflows_dir = os.path.join(data_dir, "workflows")
        os.makedirs(self.workflows_dir, exist_ok=True)
        
        self.components = components
        self.active_workflows: Dict[str, Workflow] = {}
        
        # Task executors
        self.task_executors: Dict[TaskType, Callable] = {
            TaskType.DATA_INGESTION: self._execute_data_ingestion,
            TaskType.EMBEDDING_GENERATION: self._execute_embedding_generation,
            TaskType.DATA_ANALYSIS: self._execute_data_analysis,
            TaskType.INSIGHT_GENERATION: self._execute_insight_generation,
            TaskType.VISUALIZATION: self._execute_visualization,
            TaskType.REPORT_GENERATION: self._execute_report_generation,
            TaskType.CUSTOM: self._execute_custom_task
        }
        
        # Load existing workflows
        self._load_workflows()
    
    def create_workflow(self, name: str, description: str, tasks: List[Dict[str, Any]]) -> str:
        """
        Create a new workflow
        
        Args:
            name: Workflow name
            description: Workflow description
            tasks: List of task definitions
            
        Returns:
            Workflow ID
        """
        workflow_id = str(uuid.uuid4())
        
        # Create workflow tasks
        workflow_tasks = []
        for task_def in tasks:
            task = WorkflowTask(
                id=str(uuid.uuid4()),
                name=task_def['name'],
                task_type=TaskType(task_def['task_type']),
                dependencies=task_def.get('dependencies', []),
                parameters=task_def.get('parameters', {})
            )
            workflow_tasks.append(task)
        
        # Create workflow
        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description,
            tasks=workflow_tasks
        )
        
        # Save workflow
        self._save_workflow(workflow)
        self.active_workflows[workflow_id] = workflow
        
        logger.info(f"Created workflow: {name} ({workflow_id})")
        return workflow_id
    
    def execute_workflow(self, workflow_id: str) -> bool:
        """
        Execute a workflow
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Success status
        """
        if workflow_id not in self.active_workflows:
            logger.error(f"Workflow not found: {workflow_id}")
            return False
        
        workflow = self.active_workflows[workflow_id]
        
        try:
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_at = datetime.now()
            
            logger.info(f"Starting workflow execution: {workflow.name}")
            
            # Execute tasks in dependency order
            executed_tasks = set()
            
            while len(executed_tasks) < len(workflow.tasks):
                # Find tasks ready to execute
                ready_tasks = []
                for task in workflow.tasks:
                    if (task.id not in executed_tasks and 
                        task.status == WorkflowStatus.PENDING and
                        all(dep_id in executed_tasks for dep_id in task.dependencies)):
                        ready_tasks.append(task)
                
                if not ready_tasks:
                    # Check if there are failed tasks
                    failed_tasks = [t for t in workflow.tasks if t.status == WorkflowStatus.FAILED]
                    if failed_tasks:
                        workflow.status = WorkflowStatus.FAILED
                        logger.error(f"Workflow failed due to task failures: {[t.name for t in failed_tasks]}")
                        break
                    else:
                        logger.error("No ready tasks found - possible circular dependency")
                        workflow.status = WorkflowStatus.FAILED
                        break
                
                # Execute ready tasks
                for task in ready_tasks:
                    success = self._execute_task(task)
                    executed_tasks.add(task.id)
                    
                    if not success:
                        workflow.status = WorkflowStatus.FAILED
                        logger.error(f"Task failed: {task.name}")
                        break
                
                if workflow.status == WorkflowStatus.FAILED:
                    break
            
            # Check final status
            if workflow.status != WorkflowStatus.FAILED:
                workflow.status = WorkflowStatus.COMPLETED
                workflow.completed_at = datetime.now()
                logger.info(f"Workflow completed successfully: {workflow.name}")
            
            # Save updated workflow
            self._save_workflow(workflow)
            return workflow.status == WorkflowStatus.COMPLETED
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            logger.error(f"Error executing workflow {workflow.name}: {e}")
            self._save_workflow(workflow)
            return False
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get workflow status
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Workflow status information
        """
        if workflow_id not in self.active_workflows:
            return None
        
        workflow = self.active_workflows[workflow_id]
        return {
            'id': workflow.id,
            'name': workflow.name,
            'status': workflow.status.value,
            'progress': self._calculate_progress(workflow),
            'tasks': [
                {
                    'id': task.id,
                    'name': task.name,
                    'status': task.status.value,
                    'error': task.error
                }
                for task in workflow.tasks
            ]
        }
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """
        List all workflows
        
        Returns:
            List of workflow summaries
        """
        return [
            {
                'id': workflow.id,
                'name': workflow.name,
                'description': workflow.description,
                'status': workflow.status.value,
                'created_at': workflow.created_at.isoformat(),
                'task_count': len(workflow.tasks)
            }
            for workflow in self.active_workflows.values()
        ]
    
    def create_data_analysis_workflow(self, dataset_path: str, analysis_type: str = "comprehensive") -> str:
        """
        Create a predefined data analysis workflow
        
        Args:
            dataset_path: Path to the dataset
            analysis_type: Type of analysis (comprehensive, quick, custom)
            
        Returns:
            Workflow ID
        """
        if analysis_type == "comprehensive":
            tasks = [
                {
                    'name': 'Ingest Dataset',
                    'task_type': 'data_ingestion',
                    'parameters': {'file_path': dataset_path}
                },
                {
                    'name': 'Generate Embeddings',
                    'task_type': 'embedding_generation',
                    'dependencies': ['Ingest Dataset'],
                    'parameters': {}
                },
                {
                    'name': 'Exploratory Data Analysis',
                    'task_type': 'data_analysis',
                    'dependencies': ['Ingest Dataset'],
                    'parameters': {'analysis_type': 'exploratory'}
                },
                {
                    'name': 'Statistical Analysis',
                    'task_type': 'data_analysis',
                    'dependencies': ['Exploratory Data Analysis'],
                    'parameters': {'analysis_type': 'statistical'}
                },
                {
                    'name': 'Generate Insights',
                    'task_type': 'insight_generation',
                    'dependencies': ['Statistical Analysis', 'Generate Embeddings'],
                    'parameters': {}
                },
                {
                    'name': 'Create Visualizations',
                    'task_type': 'visualization',
                    'dependencies': ['Statistical Analysis'],
                    'parameters': {}
                },
                {
                    'name': 'Generate Report',
                    'task_type': 'report_generation',
                    'dependencies': ['Generate Insights', 'Create Visualizations'],
                    'parameters': {}
                }
            ]
        elif analysis_type == "quick":
            tasks = [
                {
                    'name': 'Ingest Dataset',
                    'task_type': 'data_ingestion',
                    'parameters': {'file_path': dataset_path}
                },
                {
                    'name': 'Quick Analysis',
                    'task_type': 'data_analysis',
                    'dependencies': ['Ingest Dataset'],
                    'parameters': {'analysis_type': 'quick'}
                },
                {
                    'name': 'Generate Summary',
                    'task_type': 'insight_generation',
                    'dependencies': ['Quick Analysis'],
                    'parameters': {'summary_only': True}
                }
            ]
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
        return self.create_workflow(
            name=f"Data Analysis - {os.path.basename(dataset_path)}",
            description=f"{analysis_type.title()} analysis of {dataset_path}",
            tasks=tasks
        )
    
    def _execute_task(self, task: WorkflowTask) -> bool:
        """Execute a single task"""
        try:
            task.status = WorkflowStatus.RUNNING
            task.start_time = datetime.now()
            
            logger.info(f"Executing task: {task.name}")
            
            # Get executor for task type
            executor = self.task_executors.get(task.task_type)
            if not executor:
                raise ValueError(f"No executor found for task type: {task.task_type}")
            
            # Execute task
            result = executor(task)
            
            task.result = result
            task.status = WorkflowStatus.COMPLETED
            task.end_time = datetime.now()
            
            logger.info(f"Task completed: {task.name}")
            return True
            
        except Exception as e:
            task.status = WorkflowStatus.FAILED
            task.error = str(e)
            task.end_time = datetime.now()
            logger.error(f"Task failed: {task.name} - {e}")
            return False
    
    def _execute_data_ingestion(self, task: WorkflowTask) -> Any:
        """Execute data ingestion task"""
        data_ingestion = self.components.get('data_ingestion')
        if not data_ingestion:
            raise ValueError("Data ingestion component not available")
        
        file_path = task.parameters.get('file_path')
        if not file_path:
            raise ValueError("file_path parameter required for data ingestion")
        
        # Determine file type and ingest
        if file_path.endswith('.csv'):
            return data_ingestion.ingest_csv(file_path)
        elif file_path.endswith('.json'):
            return data_ingestion.ingest_json(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            return data_ingestion.ingest_excel(file_path)
        else:
            return data_ingestion.ingest_text(file_path)
    
    def _execute_embedding_generation(self, task: WorkflowTask) -> Any:
        """Execute embedding generation task"""
        vector_embedding = self.components.get('vector_embedding')
        if not vector_embedding:
            raise ValueError("Vector embedding component not available")
        
        dataset_id = task.parameters.get('dataset_id')
        if not dataset_id:
            # Try to get from previous task result
            for prev_task in self.active_workflows[task.id].tasks:
                if prev_task.task_type == TaskType.DATA_INGESTION and prev_task.result:
                    dataset_id = prev_task.result
                    break
        
        if not dataset_id:
            raise ValueError("dataset_id not found for embedding generation")
        
        success = vector_embedding.generate_embeddings(dataset_id)
        if not success:
            raise ValueError("Failed to generate embeddings")
        
        return dataset_id
    
    def _execute_data_analysis(self, task: WorkflowTask) -> Any:
        """Execute data analysis task"""
        # This would integrate with your data analysis components
        analysis_type = task.parameters.get('analysis_type', 'exploratory')
        
        # Placeholder for actual analysis implementation
        logger.info(f"Performing {analysis_type} analysis")
        
        return {
            'analysis_type': analysis_type,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
    
    def _execute_insight_generation(self, task: WorkflowTask) -> Any:
        """Execute insight generation task"""
        model_wrapper = self.components.get('model_wrapper')
        if not model_wrapper:
            raise ValueError("Model wrapper component not available")
        
        # Generate insights using the AI model
        prompt = "Generate insights from the analyzed data"
        insights = model_wrapper.generate_response(prompt)
        
        return {
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        }
    
    def _execute_visualization(self, task: WorkflowTask) -> Any:
        """Execute visualization task"""
        # Placeholder for visualization generation
        logger.info("Creating visualizations")
        
        return {
            'visualizations': ['chart1.png', 'chart2.png'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _execute_report_generation(self, task: WorkflowTask) -> Any:
        """Execute report generation task"""
        # Placeholder for report generation
        logger.info("Generating report")
        
        return {
            'report_path': 'analysis_report.pdf',
            'timestamp': datetime.now().isoformat()
        }
    
    def _execute_custom_task(self, task: WorkflowTask) -> Any:
        """Execute custom task"""
        # Placeholder for custom task execution
        logger.info(f"Executing custom task: {task.name}")
        
        return {
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_progress(self, workflow: Workflow) -> float:
        """Calculate workflow progress percentage"""
        if not workflow.tasks:
            return 0.0
        
        completed_tasks = sum(1 for task in workflow.tasks if task.status == WorkflowStatus.COMPLETED)
        return (completed_tasks / len(workflow.tasks)) * 100
    
    def _save_workflow(self, workflow: Workflow):
        """Save workflow to disk"""
        workflow_path = os.path.join(self.workflows_dir, f"{workflow.id}.json")
        with open(workflow_path, 'w') as f:
            json.dump(workflow.to_dict(), f, indent=2)
    
    def _load_workflows(self):
        """Load existing workflows from disk"""
        if not os.path.exists(self.workflows_dir):
            return
        
        for filename in os.listdir(self.workflows_dir):
            if filename.endswith('.json'):
                try:
                    workflow_path = os.path.join(self.workflows_dir, filename)
                    with open(workflow_path, 'r') as f:
                        workflow_data = json.load(f)
                    
                    workflow = Workflow.from_dict(workflow_data)
                    self.active_workflows[workflow.id] = workflow
                    
                except Exception as e:
                    logger.error(f"Error loading workflow {filename}: {e}")