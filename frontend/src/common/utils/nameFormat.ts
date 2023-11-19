export const isNameValid = (name: string) : boolean => {
    return /[^a-zA-Z]/.test(name)
}