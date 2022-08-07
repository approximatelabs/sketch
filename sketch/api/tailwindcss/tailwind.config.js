/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["../templates/**/*.{html,js}"],
  theme: {
    extend: {},
  },
  plugins: [],
  darkMode: 'class',
}

// consider breaking keyframes and animations (theme.extend.animation, theme.extend.keyframes) into separate files
// create a local library of language for "animations", and then just use them in the elements. 