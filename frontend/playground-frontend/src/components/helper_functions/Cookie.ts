export const setCookie = (
  cname: string,
  cvalue: string | number | boolean | null,
  exdays = 1,
  path = "/"
) => {
  const d = new Date();
  d.setTime(d.getTime() + exdays * 24 * 60 * 60 * 1000);
  document.cookie = `${cname.trim()}=${cvalue};expires=${d.toUTCString()};path=${path}`;
};

export const getCookie = (cname: string) => {
  const ca = document.cookie.split(";");
  for (let i = 0; i < ca.length; i++) {
    const c = ca[i];
    if (c.trim().substring(0, cname.length) === cname) {
      return c.substring(cname.length + 2, c.length);
    }
  }
  return undefined;
};

export const deleteCookie = (cname: string, path = "") => {
  if (getCookie(cname)) {
    document.cookie = `${cname.trim()}=;path=${path};expires=Thu, 01 Jan 1970 00:00:01 GMT`;
  }
};
