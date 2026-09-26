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
    // Props are documented with JSDoc on every component (e.g.
    // CaptionBand.jsx). Enforcing prop-types would add the prop-types runtime
    // dependency for checks nothing reads. Turned off deliberately: until
    // 2026-09-26 the lint script never linted .jsx files at all (ESLint 8
    // only picks up .js by default without --ext), which hid this rule
    // firing 42 times.
    "react/prop-types": "off",
  },
};
