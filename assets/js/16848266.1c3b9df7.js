"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[7298],{3905:(e,t,r)=>{r.d(t,{Zo:()=>u,kt:()=>d});var n=r(7294);function a(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function f(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function o(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?f(Object(r),!0).forEach((function(t){a(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):f(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function i(e,t){if(null==e)return{};var r,n,a=function(e,t){if(null==e)return{};var r,n,a={},f=Object.keys(e);for(n=0;n<f.length;n++)r=f[n],t.indexOf(r)>=0||(a[r]=e[r]);return a}(e,t);if(Object.getOwnPropertySymbols){var f=Object.getOwnPropertySymbols(e);for(n=0;n<f.length;n++)r=f[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var l=n.createContext({}),c=function(e){var t=n.useContext(l),r=t;return e&&(r="function"==typeof e?e(t):o(o({},t),e)),r},u=function(e){var t=c(e.components);return n.createElement(l.Provider,{value:t},e.children)},s="mdxType",p={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},m=n.forwardRef((function(e,t){var r=e.components,a=e.mdxType,f=e.originalType,l=e.parentName,u=i(e,["components","mdxType","originalType","parentName"]),s=c(r),m=a,d=s["".concat(l,".").concat(m)]||s[m]||p[m]||f;return r?n.createElement(d,o(o({ref:t},u),{},{components:r})):n.createElement(d,o({ref:t},u))}));function d(e,t){var r=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var f=r.length,o=new Array(f);o[0]=m;var i={};for(var l in t)hasOwnProperty.call(t,l)&&(i[l]=t[l]);i.originalType=e,i[s]="string"==typeof e?e:a,o[1]=i;for(var c=2;c<f;c++)o[c]=r[c];return n.createElement.apply(null,o)}return n.createElement.apply(null,r)}m.displayName="MDXCreateElement"},2654:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>l,contentTitle:()=>o,default:()=>p,frontMatter:()=>f,metadata:()=>i,toc:()=>c});var n=r(7462),a=(r(7294),r(3905));const f={id:"fft.fft"},o=void 0,i={unversionedId:"API/fft.fft",id:"API/fft.fft",title:"fft.fft",description:"fft.fft",source:"@site/docs/API/fft.fft.md",sourceDirName:"API",slug:"/API/fft.fft",permalink:"/fmmax/API/fft.fft",draft:!1,editUrl:"https://github.com/facebookresearch/fmmax/docs/API/fft.fft.md",tags:[],version:"current",frontMatter:{id:"fft.fft"},sidebar:"APISidebar",previous:{title:"farfield.unflatten_transverse_wavevectors",permalink:"/fmmax/API/farfield.unflatten_transverse_wavevectors"},next:{title:"fft.fourier_convolution_matrix",permalink:"/fmmax/API/fft.fourier_convolution_matrix"}},l={},c=[{value:"<code>fft.fft</code>",id:"fftfft",level:3},{value:"Args:",id:"args",level:4},{value:"Returns:",id:"returns",level:4}],u={toc:c},s="wrapper";function p(e){let{components:t,...r}=e;return(0,a.kt)(s,(0,n.Z)({},u,r,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h3",{id:"fftfft"},(0,a.kt)("inlineCode",{parentName:"h3"},"fft.fft")),(0,a.kt)("p",null,"Returns the 2D Fourier transform of ",(0,a.kt)("inlineCode",{parentName:"p"},"x"),"."),(0,a.kt)("h4",{id:"args"},"Args:"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"x"),": The array to be transformed."),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"expansion"),": The field expansion to be used."),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"axes"),": The axes to be transformed, with default being ",(0,a.kt)("inlineCode",{parentName:"li"},"(-2, -1)"),", the final axes.")),(0,a.kt)("h4",{id:"returns"},"Returns:"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"None"),": The transformed ",(0,a.kt)("inlineCode",{parentName:"li"},"x"),".")))}p.isMDXComponent=!0}}]);