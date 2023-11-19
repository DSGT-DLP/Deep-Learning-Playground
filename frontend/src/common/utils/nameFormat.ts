export const isNameValid = (name: string) : boolean => {
    name = name.toLowerCase();
    for (let i = 0; i < name.length; ++i) {
        const c = name.charAt(i);
        if (c.toLowerCase() === c.toUpperCase()) {
            return false;
        }
    }
    return true;
};
