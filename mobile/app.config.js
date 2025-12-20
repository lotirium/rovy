const baseConfig = require("./app.json");
const withAndroidBuildGradleTweaks = require("./plugins/withAndroidBuildGradleTweaks");

const LOCAL_PLUGIN_MAP = {
  "./plugins/withAndroidBuildGradleTweaks": withAndroidBuildGradleTweaks,
};

module.exports = () => {
  const config = { ...baseConfig };
  const expoConfig = { ...(config.expo ?? {}) };
  const plugins = Array.isArray(expoConfig.plugins) ? [...expoConfig.plugins] : [];

  expoConfig.plugins = plugins.map((plugin) => {
    if (typeof plugin === "string" && LOCAL_PLUGIN_MAP[plugin]) {
      return LOCAL_PLUGIN_MAP[plugin];
    }

    return plugin;
  });

  config.expo = expoConfig;
  return config;
};
