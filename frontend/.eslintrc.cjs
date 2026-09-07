module.exports = {
  root: true,
  env: { browser: true, es2021: true, node: true },
  extends: [
    "eslint:recommended",
    "plugin:react/recommended",
    "plugin:react-hooks/recommended",
    "prettier",
  ],
  parserOptions: {
    ecmaVersion: "latest",
    sourceType: "module",
    ecmaFeatures: { jsx: true },
  },
  settings: { react: { version: "detect" } },
  rules: {
    // Functional components only -- no class components.
    "no-restricted-syntax": [
      "error",
      {
        selector:
          "ClassDeclaration[superClass.object.name='React'], ClassDeclaration[superClass.name='Component'], ClassDeclaration[superClass.name='PureComponent']",
        message: "Class components are not allowed. Use functional components with hooks.",
      },
    ],
    "react/react-in-jsx-scope": "off",
  },
};
