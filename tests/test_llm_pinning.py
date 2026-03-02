import unittest
from unittest.mock import MagicMock, patch
from polyquant.utils.llm_client import call_llm_json
from polyquant.utils.config import config

class TestLLMPinning(unittest.TestCase):
    def setUp(self):
        # Save original config values
        self.original_model = config.llm_model
        self.original_temp = config.llm_temperature

    def tearDown(self):
        # Restore original config values
        config.llm_model = self.original_model
        config.llm_temperature = self.original_temp

    @patch("polyquant.utils.llm_client.get_llm_client")
    def test_default_model_from_config(self, mock_get_client):
        # Setup mock client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"result": "success"}'))]
        mock_client.chat.completions.create.return_value = mock_response

        # 1. Test with default config value
        config.llm_model = "test/default-model"
        call_llm_json("Hello")
        
        # Verify call used the config model
        mock_client.chat.completions.create.assert_called()
        args, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], "test/default-model")

    @patch("polyquant.utils.llm_client.get_llm_client")
    def test_explicit_model_override(self, mock_get_client):
        # Setup mock client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"result": "success"}'))]
        mock_client.chat.completions.create.return_value = mock_response

        # 2. Test explicit override in function call
        config.llm_model = "test/wrong-model"
        call_llm_json("Hello", model="test/pinned-model")
        
        # Verify call used the explicit model, not the config one
        args, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], "test/pinned-model")

    @patch("polyquant.utils.llm_client.get_llm_client")
    def test_temperature_pinning(self, mock_get_client):
        # Setup mock client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"result": "success"}'))]
        mock_client.chat.completions.create.return_value = mock_response

        # 3. Test temperature pinning
        config.llm_temperature = 0.88
        call_llm_json("Hello")
        
        # Verify call used the config temperature
        args, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["temperature"], 0.88)

if __name__ == "__main__":
    unittest.main()
