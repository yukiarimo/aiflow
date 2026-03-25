import os
import yaml
from utils import Yuna_Weather, Yuna_News, Yuna_Email, Yuna_Calendar


#  THE ENGINE
class LoliConnect:
	def __init__(self, config_path="loliconnect_test.yaml"):
		self.config_path = config_path
		self.data = self._load_yaml()

		# Initialize Tools
		self.tools = {"weather": Yuna_Weather(), "kagi_news": Yuna_News(), "email": Yuna_Email(), "calendar": Yuna_Calendar(), }

		# Map YAML script names to Python Methods
		self.script_map = {
		    "weather": self.tools["weather"].get_forecast,
		    "kagi_news": self.tools["kagi_news"].get_news,
		    # Add others as needed
		}

	def _load_yaml(self):
		if not os.path.exists(self.config_path):
			raise FileNotFoundError(f"Config {self.config_path} not found.")
		with open(self.config_path, "r") as f:
			return yaml.safe_load(f)

	def resolve_var(self, value):
		"""Resolves $variables.xyz"""
		if isinstance(value, str) and value.startswith("$variables."):
			key = value.split(".")[1]
			return self.data.get("variables", {}).get(key, value)
		return value

	def run_chain(self, chain_name, user_inputs=None):
		"""Executes a chain defined in YAML"""
		if user_inputs is None:
			user_inputs = {}

		chain_def = self.data.get("chains", {}).get(chain_name)
		if not chain_def:
			return f"❌ Chain '{chain_name}' not found."

		print(f"🔗 Starting Chain: {chain_name}")

		# 1. Parse Arguments & Defaults
		args_def = chain_def.get("args", {})
		context = {}  # Store step outputs here

		final_inputs = {}
		for arg_key, arg_cfg in args_def.items():
			# Priority: User Input > YAML Default > Variable
			val = user_inputs.get(arg_key)
			if val is None:
				val = self.resolve_var(arg_cfg.get("default"))
			final_inputs[arg_key] = val

		context["input"] = final_inputs

		# 2. Execute Flow
		flow = chain_def.get("flow", [])
		for step in flow:
			step_id = step["id"]
			step_type = step["type"]
			resource = step["resource"]

			print(f"  ➡️ Step: {step_id} ({step_type})")

			# Resolve Inputs for this step
			step_inputs = {}
			for k, v in step.get("inputs", {}).items():
				# Handle $input.x
				if isinstance(v, str) and v.startswith("$input."):
					step_inputs[k] = context["input"].get(v.split(".")[1])
				# Handle $step.id.output
				elif isinstance(v, str) and v.startswith("$step."):
					parts = v.split(".")  # $step, weather_data, output
					ref_id = parts[1]
					step_inputs[k] = context.get(ref_id, {}).get("output")
				else:
					step_inputs[k] = v

			# EXECUTE
			output = None

			if step_type == "script":
				# Find the actual function name from YAML mapping
				# $scripts.weather_fetcher -> "weather" -> self.script_map["weather"]
				script_key = resource.replace("$scripts.", "")
				real_func_name = self.data["scripts"].get(script_key)
				func = self.script_map.get(real_func_name)

				if func:
					# Python magic to pass dict as kwargs
					# Note: You might need to adjust Yuna_News.get_news signature to match YAML keys (category vs topic)
					try:
						output = func(**step_inputs)
					except TypeError as e:
						output = f"Arg Error: {e}"
				else:
					output = "Function not implemented."

			elif step_type == "loli":
				# Load the Loli Prompt template
				# resource = "lolis/weather_narrator.txt" (resolved from $lolis)
				loli_key = resource.replace("$lolis.", "")
				prompt_path = self.data["lolis"].get(loli_key)
				# In a real app, you'd load the text file and inject variables
				# For now, we simulate the "Loli" processing
				output = (f"[LOLI GENERATION]\nPrompt: {prompt_path}\nContext: {step_inputs}")

			elif step_type == "chain":
				# Recursive Chain!
				# resource = "weather_forecast"
				output = self.run_chain(resource, step_inputs)

			# Store Output
			context[step_id] = {"output": output}

			if step.get("output") == "return":
				return output

		return context
