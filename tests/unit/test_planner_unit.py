"""Unit tests for Planner with mocked LLM."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent.config.loader import AgentConfig
from agent.runtime.planner import Planner
from agent.runtime.schema import Plan, PlanStep, ToolName, InvalidPlanError, JSONExtractionError


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self.model = "test-model"

    async def chat(self, messages, **kwargs):
        """Mock chat method."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return '{"plan_id": "test-plan", "objective": "Test", "steps": []}'

    async def generate(self, prompt, **kwargs):
        """Mock generate method."""
        return await self.chat([{"role": "user", "content": prompt}], **kwargs)


class TestPlannerUnit:
    """Unit tests for Planner with mocked LLM."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return AgentConfig()

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        return MockLLMProvider()

    @pytest.fixture
    def planner(self, config, mock_llm):
        """Create planner with mock LLM."""
        return Planner(config, mock_llm)

    def test_planner_initialization(self, planner, config, mock_llm):
        """Test planner initialization."""
        assert planner.config == config
        assert planner.llm == mock_llm
        assert planner.model_name == "test-model"

    def test_build_system_prompt(self, planner):
        """Test system prompt building."""
        prompt = planner._build_system_prompt()
        assert "planning assistant" in prompt.lower()
        assert "tools" in prompt.lower()
        assert "filesystem" in prompt.lower() or "system" in prompt.lower()

    def test_extract_json_from_markdown(self, planner):
        """Test extracting JSON from markdown code block."""
        text = 'Here is the plan:\n```json\n{"plan_id": "test", "steps": []}\n```'
        json_str = planner._extract_json(text)
        assert json_str == '{"plan_id": "test", "steps": []}'

    def test_extract_json_direct(self, planner):
        """Test extracting JSON directly."""
        text = 'Some text {"plan_id": "test", "steps": []} more text'
        json_str = planner._extract_json(text)
        assert "plan_id" in json_str
        assert "test" in json_str

    def test_extract_json_multiple_objects_error(self, planner):
        """Test error when multiple JSON objects found."""
        text = '{"one": 1} {"two": 2}'
        with pytest.raises(JSONExtractionError):
            planner._extract_json(text)

    def test_extract_json_no_json_error(self, planner):
        """Test error when no JSON found."""
        text = "No JSON here at all"
        with pytest.raises(JSONExtractionError):
            planner._extract_json(text)

    def test_parse_and_validate_plan_valid(self, planner):
        """Test parsing valid plan."""
        json_str = '{"plan_id": "test", "objective": "Test", "steps": []}'
        plan = planner._parse_and_validate_plan(json_str)
        assert isinstance(plan, Plan)
        assert plan.plan_id == "test"
        assert plan.objective == "Test"

    def test_parse_and_validate_plan_invalid_json(self, planner):
        """Test parsing invalid JSON."""
        json_str = '{"invalid": json}'
        with pytest.raises(InvalidPlanError):
            planner._parse_and_validate_plan(json_str)

    def test_parse_and_validate_plan_invalid_schema(self, planner):
        """Test parsing JSON with invalid schema."""
        json_str = '{"invalid": "data"}'
        with pytest.raises(InvalidPlanError):
            planner._parse_and_validate_plan(json_str)

    def test_generate_plan_id_deterministic(self, planner):
        """Test that plan ID is deterministic for same input."""
        goal = "Test goal"
        id1 = planner._generate_plan_id(goal)
        id2 = planner._generate_plan_id(goal)
        # Should be same if generated in same minute
        assert id1 == id2

    def test_generate_plan_id_different_goals(self, planner):
        """Test that different goals produce different IDs."""
        id1 = planner._generate_plan_id("Goal 1")
        id2 = planner._generate_plan_id("Goal 2")
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_plan_success(self, planner, mock_llm):
        """Test successful planning."""
        mock_llm.responses = ['{"plan_id": "test-plan", "objective": "Test", "steps": []}']
        
        result = await planner.plan("Test goal")
        
        assert result.plan.plan_id == "test-plan"
        assert result.plan.objective == "Test"
        assert result.diagnostics.model_name == "test-model"

    @pytest.mark.asyncio
    async def test_plan_with_steps(self, planner, mock_llm):
        """Test planning with steps."""
        plan_json = '''{
            "plan_id": "test-plan",
            "objective": "Test",
            "steps": [
                {
                    "step_id": 1,
                    "tool": "filesystem",
                    "operation": "read_file",
                    "arguments": {"path": "test.txt"},
                    "rationale": "Read file"
                }
            ]
        }'''
        mock_llm.responses = [f'```json\n{plan_json}\n```']
        
        result = await planner.plan("Test goal")
        
        assert len(result.plan.steps) == 1
        assert result.plan.steps[0].tool == ToolName.FILESYSTEM
        assert result.plan.steps[0].operation == "read_file"

    @pytest.mark.asyncio
    async def test_plan_llm_error(self, planner, mock_llm):
        """Test planning when LLM raises error."""
        async def failing_chat(*args, **kwargs):
            raise Exception("LLM error")
        
        mock_llm.chat = failing_chat
        
        result = await planner.plan("Test goal")
        
        assert result.plan is None
        assert result.error is not None
        assert "LLM error" in result.error

    @pytest.mark.asyncio
    async def test_plan_json_extraction_error(self, planner, mock_llm):
        """Test planning when JSON extraction fails."""
        mock_llm.responses = ["No JSON here"]
        
        result = await planner.plan("Test goal")
        
        assert result.plan is None
        assert result.error is not None
        assert "JSON" in result.error or "extraction" in result.error.lower()

    @pytest.mark.asyncio
    async def test_plan_validation_error(self, planner, mock_llm):
        """Test planning when validation fails."""
        mock_llm.responses = ['{"invalid": "plan"}']
        
        result = await planner.plan("Test goal")
        
        assert result.plan is None
        assert result.error is not None
        assert "validation" in result.error.lower() or "invalid" in result.error.lower()
